import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint
# from utils.metric_calc import *
from src.utils.metric_calc import * 
class ODEFunc(nn.Module):
    def __init__(self, dim, w=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, w),
            nn.SiLU(),
            nn.LayerNorm(w),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.LayerNorm(w),
            nn.Linear(w, dim),
        )

    def forward(self, t, x):
        return self.net(x)

class ODEBaseline(nn.Module):
    def __init__(self, dim=2, w=64):
        super().__init__()
        self.func = ODEFunc(dim, w)
    def forward(self, x0, t1):
        t_span = torch.stack([torch.tensor(0., device=x0.device), t1])
        return odeint(self.func, x0, t_span)[-1]
    
    @torch.no_grad()
    def predict_autoreg(self, x0, dt=0.1, F=8, rtol=1e-5, atol=1e-7):
        preds = []
        preds.append(x0)
        t_step = torch.tensor([0.0, dt], device=x0.device, dtype=x0.dtype)
        for _ in range(F):
            x = odeint(self.func, preds[-1], t_step, rtol=rtol, atol=atol)[-1]  # 0→dt
            preds.append(x.detach())
        return torch.stack(preds[1:], dim=1)
    
    @torch.no_grad()
    def predict_horizons(self, x0, dt: float, F: int, rtol=1e-6, atol=1e-9):
        times = torch.linspace(0.0, float(F*dt), steps=F+1, device=x0.device, dtype=x0.dtype)
        traj = odeint(self.func, x0, times, rtol=rtol, atol=atol)  # (F+1, N, 2)
        return traj[1:]

class ODEBaseline_lighting(pl.LightningModule):
    def __init__(self, 
                 dim=2, 
                 w=64, 
                 lr=1e-5, 
                 loss_fn=nn.MSELoss(),
                 metrics = ['mse_loss', 'l1_loss']):
        super().__init__()
        self.ode_func = ODEFunc(dim, w)
        self.lr = lr
        self.loss_fn = loss_fn
        self.naming = 'ODEBaseline'
        self.metrics = metrics

    def forward(self, x0, t_span):
        return odeint(self.ode_func, x0, t_span)

    def training_step(self, batch, batch_idx):
        """x0, x0_class, x1, x0_time, x1_time """
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze() #- x0_time
        print("training_step")
        # print(x0.shape, x1.shape, t_span.shape) # torch.Size([256, 2]) torch.Size([256, 2]) torch.Size([256])
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())

        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_val', v)

        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x0, x0_class, x1, x0_time, x1_time = batch
        t_span = x1_time.squeeze()
        x_pred = self.forward(x0, t_span)
        loss = self.loss_fn(x_pred[-1], x1.squeeze())
        # metrics
        metricsD = metrics_calculation(x_pred[-1], x1, metrics=self.metrics)
        for k, v in metricsD.items():
            self.log(f'{k}_test', v)

        self.log('test_loss', loss)
        return loss
