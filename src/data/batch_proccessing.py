import torch
def agentbatch_to_modelpack(agent_batch, memory=4, dim=2, future=8):
    G = (lambda n: agent_batch[n]) if isinstance(agent_batch, dict) else (lambda n: getattr(agent_batch, n))

    hist = G("agent_hist")          # (B,A,H,2)
    fut  = G("agent_fut")           # (B,A,Ffull,2)
    hlen = G("agent_hist_len")      # (B,A)
    flen = G("agent_fut_len")       # (B,A)
    dt   = G("dt")                  # (B,)

    B, A, H, _ = hist.shape
    Fneed = min(future, fut.size(2))

    valid = (hlen >= (memory + 1)) & (flen >= Fneed)
    if valid.sum() == 0:
        return None

    b_idx, a_idx = torch.nonzero(valid, as_tuple=True)             # (N,)
    t0_idx = (hlen - 1).clamp(min=0)[b_idx, a_idx]                 # (N,)

    # x0: текущее состояние
    x0 = hist[b_idx, a_idx, t0_idx, :dim]                          # (N,dim)

    # x1: вся будущая траектория длиной Fneed
    x1_full = fut[b_idx, a_idx, :Fneed, :dim]                      # (N,Fneed,dim)
    x1_next = x1_full[:, 0, :]                                     # (N,dim) — первый шаг, удобно для обучения

    # history -> флет (N, 2*memory)
    M = memory
    k = torch.arange(1, M + 1, device=hist.device)
    t_hist_idx = (t0_idx.unsqueeze(1) - k.unsqueeze(0)).clamp(0, H - 1)
    t_hist_idx = torch.flip(t_hist_idx, dims=[1])
    hist_sel = hist[b_idx.unsqueeze(1), a_idx.unsqueeze(1), t_hist_idx, :dim]  # (N,M,dim)
    x0_class = hist_sel.reshape(hist_sel.shape[0], -1)                          # (N,M*dim)

    # Remove NaN
    nan_mask = (
        torch.isnan(x0).any(dim=1) |
        torch.isnan(x0_class).any(dim=1) |
        torch.isnan(x1_full).view(x1_full.size(0), -1).any(dim=1)
    )
    keep = ~nan_mask
    if keep.sum() == 0:
        return None

    x0       = x0[keep].to(torch.float32)
    x0_class = x0_class[keep].to(torch.float32)
    x1_full  = x1_full[keep].to(torch.float32)
    x1_next  = x1_next[keep].to(torch.float32)

    t0 = torch.zeros(x0.size(0), dtype=hist.dtype, device=hist.device)
    t1 = dt[b_idx][keep].to(hist.dtype)  # per-sample dt

    # Возвращаем и «всё будущее», и «первый шаг»
    return x0, x0_class, x1_full, x1_next, t0, t1


def make_model_collate(dataset, memory=4, dim=2, future=8):
    base_collate = dataset.get_collate_fn()
    def collate_fn(scene_list):
        agent_batch = base_collate(scene_list)  # -> AgentBatch
        return agentbatch_to_modelpack(agent_batch, memory=memory, dim=dim, future=future)
    return collate_fn