# src/models/ode_ctx.py
import torch, torch.nn as nn
from torchdiffeq import odeint

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.LayerNorm(d), nn.Linear(d,d))
    def forward(self, h): return h + self.f(h)

class ODEFuncCtx(nn.Module):
    def __init__(self, x_dim=2, hist_len=4, ctx_dim=32, w=384, nblocks=3):
        super().__init__()
        self.ctx_mlp = nn.Sequential(
            nn.Linear(2*hist_len, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, ctx_dim), nn.SiLU(),
        )
        self.inp = nn.Sequential(nn.Linear(x_dim + 1 + ctx_dim, w), nn.SiLU(), nn.LayerNorm(w))
        self.blocks = nn.Sequential(*[ResBlock(w) for _ in range(nblocks)])
        self.out = nn.Linear(w, x_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)

    def set_context(self, x0_class_flat: torch.Tensor):
        self._ctx = self.ctx_mlp(x0_class_flat)  # (N, ctx_dim)

    def forward(self, t, x):
        tcol = torch.full((x.size(0),1), float(t), device=x.device, dtype=x.dtype)
        h = self.inp(torch.cat([x, tcol, self._ctx], dim=-1))
        h = self.blocks(h)
        return self.out(h)

class ODEBaselineCtx(nn.Module):
    def __init__(self, dim=2, hist_len=4, w=384, ctx_dim=32, nblocks=3):
        super().__init__()
        self.func = ODEFuncCtx(dim, hist_len, ctx_dim, w, nblocks)

    def _prep_ctx(self, x0_class): self.func.set_context(x0_class)

    def forward(self, x0, t1, x0_class, rtol=1e-6, atol=1e-9):
        self._prep_ctx(x0_class)
        t_span = torch.stack([torch.zeros((), device=x0.device, dtype=x0.dtype), t1])
        return odeint(self.func, x0, t_span, rtol=rtol, atol=atol)[-1]

    @torch.no_grad()
    def predict_horizons(self, x0, x0_class, dt: float, F: int, rtol=1e-6, atol=1e-9):
        self._prep_ctx(x0_class)
        times = torch.linspace(0.0, float(F*dt), steps=F+1, device=x0.device, dtype=x0.dtype)
        traj = odeint(self.func, x0, times, rtol=rtol, atol=atol)
        return traj[1:]  # (F,N,2)