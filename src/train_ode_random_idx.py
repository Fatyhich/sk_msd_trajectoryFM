# %%
import sys, os
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from trajdata import AgentBatch, UnifiedDataset

sys.path.append(os.path.abspath("../"))
from src.data.batch_proccessing import make_model_collate
from src.models.ode_baseline import ODEBaseline
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
pl.seed_everything(42)

# %%
dataset = UnifiedDataset(
        desired_data=["eupeds_eth","eupeds_hotel","eupeds_univ","eupeds_zara1","eupeds_zara2"],
        data_dirs={
            "eupeds_eth":  "../data/eth",
            "eupeds_hotel":"../data/eth",
            "eupeds_univ": "../data/eth",
            "eupeds_zara1":"../data/eth",
            "eupeds_zara2":"../data/eth",
        },
        desired_dt=0.1,
        state_format='x,y',
        obs_format='x,y',
        centric="scene",
        history_sec=(0.8,0.8),
        future_sec=(0.8,0.8),
        standardize_data=False,
    )
collate_fn = make_model_collate(dataset=dataset, memory=4, dim=2)

# %%
N = len(dataset)
n_test = int(0.1 * N)
n_train = N - n_test
train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=collate_fn, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=collate_fn, pin_memory=True)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
model = ODEBaseline(dim=2, w=512).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def ade_fde(pred, gt):  # pred, gt: (F,N,2)
    d = torch.linalg.norm(pred - gt, dim=-1)  # (F,N)
    return d.mean().item(), d[-1].mean().item()

# %%
EPOCHS = 30
F = 8  # длина будущего в collate (x1_full.shape[1])

# --- train loop
for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        if batch is None:
            continue
        x0, x0_class, x1_full, x1_next, t0, t1 = batch
        x0, x1_full, t1 = x0.to(device), x1_full.to(device), t1.to(device)

        # один k на весь батч
        k = torch.randint(1, F + 1, (1,), device=device).item()  # 1..F
        dt = t1[0].to(x0.dtype)                                  # общий dt
        t_scalar = dt * k

        # таргет — k-й шаг будущего
        target_k = x1_full[:, k - 1, :]                          # (N,2)

        optimizer.zero_grad()
        pred_k = model(x0, t_scalar)                             # (N,2)
        loss = criterion(pred_k, target_k)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x0.size(0)

    mean_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:03d}: train_loss = {mean_loss:.6f}")

    # --- валидация: авторегрессия на весь горизонт и ADE/FDE
    model.eval()
    test_loss = 0.0
    sum_ade, sum_fde, batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            x0, x0_class, x1_full, x1_next, t0, t1 = batch
            x0, x1_full, t1 = x0.to(device), x1_full.to(device), t1.to(device)

            # посчитаем MSE на случайном горизонте и ADE/FDE на полном
            k = torch.randint(1, F + 1, (1,), device=device).item()
            dt = t1[0].to(x0.dtype)
            t_scalar = dt * k
            target_k = x1_full[:, k - 1, :]

            pred_k = model(x0, t_scalar)
            test_loss += criterion(pred_k, target_k).item() * x0.size(0)

            F = x1_full.size(1)
            dt = float(t1[0].item())

            pred = model.predict_horizons(x0, dt=dt, F=F)   # (F,N,2)
            # ADE/FDE
            diff = pred - x1_full.permute(1,0,2)            # (F,N,2)
            l2 = torch.linalg.norm(diff, dim=-1)            # (F,N)
            ade = l2.mean().item()
            fde = l2[-1].mean().item()
            sum_ade += ade; sum_fde += fde; batches += 1

    test_loss /= len(test_loader.dataset)
    print(f"ADE on {epoch}: {sum_ade/batches:.6f}   FDE on {epoch}: {sum_fde/batches:.6f}")
    print(f"test_loss = {test_loss:.6f}")

print("Training finished.")
torch.save(model.state_dict(), "ode_baseline.pth")


