"""Spectral-family Muon variants for C4 benchmarking: Freon, DynMuon, SoftMuon.

Updates share the form U Sigma^p V^T (or Freon's (GG^T)^{-c} G <=> p = 1 - 2c).
SVD paths are exact references; NS / Fast-Spectral paths match the efficient paper recipes.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from muon import adam_update, zeropower_via_newtonschulz5


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _shape_scale(grad: torch.Tensor) -> float:
    return max(1, grad.size(-2) / grad.size(-1)) ** 0.5


def _momentum_update(grad, momentum, beta=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    return update


def svd_spectral_power(g: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Exact U Sigma^p V^T."""
    orig_dtype = g.dtype
    U, S, Vh = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    Sp = S.clamp_min(eps).pow(p)
    out = (U * Sp.unsqueeze(-2)) @ Vh
    return out.to(orig_dtype)


def freon_svd(g: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    """Freon: (GG^T)^{-c} G = U Sigma^{1-2c} V^T, with spectral-ball rescaling."""
    # p = 1 - 2c maps SGD (c=0,p=1) to Muon (c=1/2,p=0) to quasi-norm (c>1/2,p<0)
    p = 1.0 - 2.0 * c
    out = svd_spectral_power(g, p, eps=eps)
    # mean-Schatten / spectral ball: keep O(1) spectral step independent of c
    # scale so RMS of singular values of the update is comparable to Muon (all ones)
    m, n = g.shape[-2], g.shape[-1]
    r = float(min(m, n))
    if abs(p) > 1e-8:
        # approximate spectral normalization via Frobenius of polar factor target
        target = math.sqrt(r)
        cur = out.float().norm().clamp_min(eps)
        out = out * (target / cur)
    return out


def soft_muon_ns(g: torch.Tensor, steps: int = 5, soft_alpha: float = 0.5) -> torch.Tensor:
    """Soft-Muon: convex mix of Frobenius-normalized grad and NS Muon update.

    soft_alpha=1 recovers Muon; soft_alpha=0 is normalized SGD (underweights tiny modes less than Muon).
    """
    orig_dtype = g.dtype
    if soft_alpha >= 1.0 - 1e-12:
        return zeropower_via_newtonschulz5(g, steps=steps)
    X = g.float()
    scale = X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    Xn = X / scale
    muon = zeropower_via_newtonschulz5(Xn, steps=steps).float()
    out = (1.0 - soft_alpha) * Xn + soft_alpha * muon
    # keep polar-like scale: unit Frobenius times sqrt(rank)
    out = out / out.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    out = out * math.sqrt(float(min(g.size(-2), g.size(-1))))
    return out.to(orig_dtype)


def contra_muon_ns(g: torch.Tensor, steps: int = 5, coeff: float = 0.5) -> torch.Tensor:
    """Contra-Muon: exaggerate Muon by subtracting a fraction of operator-normalized G."""
    orig_dtype = g.dtype
    X = g.float()
    scale = X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    Xn = X / scale
    muon = zeropower_via_newtonschulz5(Xn, steps=steps).float()
    out = (1.0 + coeff) * muon - coeff * Xn
    return out.to(orig_dtype)


def fast_spectral(g: torch.Tensor, p: float, ns_steps: int = 5, eps: float = 1e-7, order: int = 2):
    """DynMuon Fast-Spectral: NS polar factor plus low-order A^{p/2} correction."""
    orig_dtype = g.dtype
    X = g.to(torch.float32)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    scale = X.norm(dim=(-2, -1), keepdim=True) + eps
    Xn = X / scale
    Y_mu = zeropower_via_newtonschulz5(Xn, steps=ns_steps).float()

    if abs(p) < 1e-12:
        U = Y_mu
    else:
        A = Xn @ Xn.mT
        m = A.size(-1)
        I = torch.eye(m, device=A.device, dtype=A.dtype).expand(A.shape)
        delta = 0.5 * p
        E = A - I
        if order == 1:
            C = I + delta * E
        else:
            E2 = E @ E
            C = I + delta * E + 0.5 * delta * (delta - 1.0) * E2
        U = C @ Y_mu
        U = U * scale.pow(p)

    if transposed:
        U = U.mT
    return U.to(orig_dtype)


def dynmuon_shape(g: torch.Tensor, p: float, ns_steps: int = 5, use_svd: bool = False):
    """DynMuon stage-wise shaping for a scheduled exponent p."""
    if p >= 0.25:
        return g
    if p >= 0.0:
        return zeropower_via_newtonschulz5(g, steps=ns_steps)
    if use_svd:
        return svd_spectral_power(g, p)
    return fast_spectral(g, p, ns_steps=ns_steps)


def logistic_p(step: int, total_steps: int, p_max=1.0, p_min=-0.25, tau=0.04, w=0.04) -> float:
    q = step / float(max(total_steps, 1))
    u = (q - tau) / max(w, 1e-8)
    a = 1.0 / (1.0 + math.exp(u))
    return p_min + a * (p_max - p_min)


class _SpectralFamilyWithAuxAdam(torch.optim.Optimizer):
    """Shared Muon+AdamW shell; subclasses set matrix_update()."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self._step = 0

    def matrix_update(self, grad, momentum, beta):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        ws, rk = _world_size(), _rank()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                pad_n = (ws - len(params) % ws) % ws
                params_pad = params + [torch.empty_like(params[-1])] * pad_n
                for base_i in range(0, len(params), ws):
                    if base_i + rk < len(params):
                        p = params[base_i + rk]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = self.matrix_update(p.grad, state["momentum_buffer"], group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    if ws > 1:
                        dist.all_gather(
                            params_pad[base_i : base_i + ws],
                            params_pad[base_i + rk],
                        )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


class FreonWithAuxAdam(_SpectralFamilyWithAuxAdam):
    """Freon with SVD update (GG^T)^{-c} G. Default c=2/3 (quasi-norm regime)."""

    def __init__(self, param_groups, freon_c: float = 2.0 / 3.0):
        super().__init__(param_groups)
        self.freon_c = freon_c

    def matrix_update(self, grad, momentum, beta):
        update = _momentum_update(grad, momentum, beta=beta)
        update = freon_svd(update, self.freon_c)
        update *= _shape_scale(grad)
        return update


class SpectralPowerWithAuxAdam(_SpectralFamilyWithAuxAdam):
    """Fixed-p spectral shaping U Sigma^p V^T (SVD). Useful for optimal-p sweeps."""

    def __init__(self, param_groups, power: float = 0.0, use_svd: bool = True):
        super().__init__(param_groups)
        self.power = power
        self.use_svd = use_svd

    def matrix_update(self, grad, momentum, beta):
        update = _momentum_update(grad, momentum, beta=beta)
        if abs(self.power) < 1e-12:
            update = zeropower_via_newtonschulz5(update, steps=5)
        elif self.use_svd:
            update = svd_spectral_power(update, self.power)
        else:
            update = fast_spectral(update, self.power)
        update *= _shape_scale(grad)
        return update


class DynMuonWithAuxAdam(_SpectralFamilyWithAuxAdam):
    """DynMuon logistic p schedule with Fast-Spectral (or SVD) shaping."""

    def __init__(
        self,
        param_groups,
        total_steps: int = 10000,
        p_max: float = 1.0,
        p_min: float = -0.25,
        tau: float = 0.04,
        width: float = 0.04,
        use_svd: bool = False,
    ):
        super().__init__(param_groups)
        self.total_steps = total_steps
        self.p_max = p_max
        self.p_min = p_min
        self.tau = tau
        self.width = width
        self.use_svd = use_svd
        self.last_p = p_max

    def matrix_update(self, grad, momentum, beta):
        self.last_p = logistic_p(
            self._step,
            self.total_steps,
            p_max=self.p_max,
            p_min=self.p_min,
            tau=self.tau,
            w=self.width,
        )
        update = _momentum_update(grad, momentum, beta=beta)
        update = dynmuon_shape(update, self.last_p, use_svd=self.use_svd)
        update *= _shape_scale(grad)
        return update


class SoftMuonWithAuxAdam(_SpectralFamilyWithAuxAdam):
    """Soft-Muon (Nilin): blend between normalized SGD and Muon NS."""

    def __init__(self, param_groups, soft_alpha: float = 0.5, ns_steps: int = 5):
        super().__init__(param_groups)
        self.soft_alpha = soft_alpha
        self.ns_steps = ns_steps

    def matrix_update(self, grad, momentum, beta):
        update = _momentum_update(grad, momentum, beta=beta)
        update = soft_muon_ns(update, steps=self.ns_steps, soft_alpha=self.soft_alpha)
        update *= _shape_scale(grad)
        return update


class ContraMuonWithAuxAdam(_SpectralFamilyWithAuxAdam):
    """Contra-Muon (Nilin): boost intermediate singular modes beyond Muon."""

    def __init__(self, param_groups, contra_coeff: float = 0.5, ns_steps: int = 5):
        super().__init__(param_groups)
        self.contra_coeff = contra_coeff
        self.ns_steps = ns_steps

    def matrix_update(self, grad, momentum, beta):
        update = _momentum_update(grad, momentum, beta=beta)
        update = contra_muon_ns(update, steps=self.ns_steps, coeff=self.contra_coeff)
        update *= _shape_scale(grad)
        return update
