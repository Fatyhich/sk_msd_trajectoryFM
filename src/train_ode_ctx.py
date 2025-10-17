# train_ode_ctx.py
import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from trajdata import UnifiedDataset

# === ваш код ===
sys.path.append(os.path.abspath("../"))
from src.data.batch_proccessing import make_model_collate   # collate с future=F
from src.models.ode_ctx import ODEBaselineCtx               # модель с контекстом

torch.manual_seed(42)

# ----------------- конфиг -----------------
DATA_ROOT = "../data/eth"
BATCH_SIZE = 256
EPOCHS = 30
F = 8                        # горизонт будущего, который кладёт collate в x1_full[:, :F]
LR = 1e-3
WD = 1e-4
CLIP_NORM = 1.0
USE_SCALE = False            # если Δ малы, поставь True
SCALE_S = 100.0              # перевод в дециметры при обучении

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- данные -----------------
dataset = UnifiedDataset(
    desired_data=["eupeds_eth","eupeds_hotel","eupeds_univ","eupeds_zara1","eupeds_zara2"],
    data_dirs={
        "eupeds_eth":DATA_ROOT, "eupeds_hotel":DATA_ROOT, "eupeds_univ":DATA_ROOT,
        "eupeds_zara1":DATA_ROOT, "eupeds_zara2":DATA_ROOT,
    },
    desired_dt=0.1,
    state_format="x,y",
    obs_format="x,y",
    centric="scene",
    history_sec=(0.8, 0.8),
    future_sec=(0.8, 0.8),
    standardize_data=False,  # можешь включить True вместо USE_SCALE
)
collate_fn = make_model_collate(dataset=dataset, memory=4, dim=2, future=F)

N = len(dataset)
n_val = int(0.1 * N)
train_ds, val_ds = random_split(dataset, [N - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=True,
                          collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True,
                          collate_fn=collate_fn)

# ----------------- модель и опт -----------------
model = ODEBaselineCtx(dim=2, hist_len=4, w=384, ctx_dim=32, nblocks=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.MSELoss()
Ks = torch.tensor([1, 2, 4, F], device=device)  # набор горизонтов для лосса

# ----------------- утилиты -----------------
def ade_fde(pred_F_N_2: torch.Tensor, gt_N_F_2: torch.Tensor):
    # pred: (F,N,2), gt: (N,F,2)
    diff = pred_F_N_2 - gt_N_F_2.permute(1,0,2)
    l2 = torch.linalg.norm(diff, dim=-1)  # (F,N)
    ade = l2.mean().item()
    fde = l2[-1].mean().item()
    return ade, fde

# ----------------- train loop -----------------
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    seen = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [train]"):
        if batch is None:
            continue
        x0, x0_class, x1_full, x1_next, t0, t1 = batch
        x0, x0_class, x1_full, t1 = x0.to(device), x0_class.to(device), x1_full.to(device), t1.to(device)
        dt = t1[0].to(x0.dtype)  # общий dt в батче

        # опциональный масштаб координат для более сильного сигнала
        if USE_SCALE:
            x0_s = x0 * SCALE_S
            x1_full_s = x1_full * SCALE_S
        else:
            x0_s = x0
            x1_full_s = x1_full

        loss = 0.0
        for k in Ks:
            pred_k = model(x0_s, dt*k, x0_class)        # (N,2)
            tgt_k  = x1_full_s[:, int(k.item())-1, :]   # (N,2)
            loss  += criterion(pred_k, tgt_k)
        loss /= len(Ks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        bs = x0.size(0)
        total_loss += loss.item() * bs
        seen += bs

    train_loss = total_loss / max(1, seen)

    # -------- validation --------
    model.eval()
    val_loss = 0.0
    seen = 0
    sum_ade, sum_fde, batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [val]"):
            if batch is None:
                continue
            x0, x0_class, x1_full, x1_next, t0, t1 = batch
            x0, x0_class, x1_full, t1 = x0.to(device), x0_class.to(device), x1_full.to(device), t1.to(device)
            dt = t1[0].to(x0.dtype)

            # масштаб как на трейне
            if USE_SCALE:
                x0_s = x0 * SCALE_S
                x1_full_s = x1_full * SCALE_S
            else:
                x0_s = x0
                x1_full_s = x1_full

            # лосс на тех же Ks
            vloss = 0.0
            for k in Ks:
                pred_k = model(x0_s, dt*k, x0_class)
                tgt_k  = x1_full_s[:, int(k.item())-1, :]
                vloss += criterion(pred_k, tgt_k)
            vloss /= len(Ks)

            # ADE/FDE на полном горизонте
            pred_full = model.predict_horizons(x0_s, x0_class, dt=float(dt.item()), F=F)  # (F,N,2)
            # если скалировали на входе — вернём масштаб для метрик
            if USE_SCALE:
                pred_full = pred_full / SCALE_S

            ade, fde = ade_fde(pred_full, x1_full)
            sum_ade += ade
            sum_fde += fde
            batches += 1

            bs = x0.size(0)
            val_loss += vloss.item() * bs
            seen += bs

    val_loss /= max(1, seen)
    mean_ade = sum_ade / max(1, batches)
    mean_fde = sum_fde / max(1, batches)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  ADE={mean_ade:.6f}  FDE={mean_fde:.6f}")

    # чекпоинт по val_loss
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "ode_ctx_best.pth")
        print("Saved: ode_ctx_best.pth")

# финальный save
torch.save(model.state_dict(), "ode_ctx_last.pth")
print("Done.")