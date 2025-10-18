import torch

import torch

def agentbatch_to_modelpack(agent_batch, memory=4, dim=2):
    def G(name):
        return agent_batch[name] if isinstance(agent_batch, dict) else getattr(agent_batch, name)

    hist = G("agent_hist")          # (B, A, H, 2)
    fut  = G("agent_fut")           # (B, A, F, 2)
    hlen = G("agent_hist_len")      # (B, A)
    flen = G("agent_fut_len")       # (B, A)
    dt   = G("dt")                  # (B,)

    B, A, H, _ = hist.shape

    valid = (hlen >= (memory + 1)) & (flen >= 1)   # (B, A)
    if valid.sum() == 0:
        return None

    b_idx, a_idx = torch.where(valid)

    hist = hist.to(torch.float32)
    fut  = fut.to(torch.float32)

    # индекс текущего момента в истории
    t0_idx = (hlen - 1).clamp(min=0)[b_idx, a_idx]  # (N,)

    # целевые точки
    x0 = hist[b_idx, a_idx, t0_idx, :dim]           # (N, dim)
    x1 = fut[b_idx, a_idx, 0, :dim]                 # (N, dim)

    # история длины `memory` до t0 (в прямом временном порядке)
    k = torch.arange(1, memory + 1, device=hist.device)              # (M,)
    t_hist_idx = (t0_idx.unsqueeze(1) - k.unsqueeze(0))              # (N, M)
    # из условия valid гарантировано t0_idx >= memory
    t_hist_idx = t_hist_idx.clamp(0, H - 1)
    t_hist_idx = torch.flip(t_hist_idx, dims=[1])                    # старые -> новые

    hist_sel = hist[b_idx.unsqueeze(1), a_idx.unsqueeze(1), t_hist_idx, :dim]  # (N, M, dim)
    x0_class = hist_sel.reshape(hist_sel.shape[0], -1)               # (N, M*dim)

    # возможные NaN в признаках/целях/интервалах
    nan_mask = (
        torch.isnan(x0).any(dim=1)
        | torch.isnan(x1).any(dim=1)
        | torch.isnan(x0_class).any(dim=1)
    )

    # также исключим, если dt по соответствующему batch-индексу NaN
    dt_sel = dt[b_idx].to(hist.dtype)                                 # (N,)
    nan_mask = nan_mask | torch.isnan(dt_sel)

    if nan_mask.any():
        keep = ~nan_mask
        if keep.sum() == 0:
            return None
        x0 = x0[keep]
        x1 = x1[keep]
        x0_class = x0_class[keep]
        b_idx = b_idx[keep]
        a_idx = a_idx[keep]
        t0_idx = t0_idx[keep]
        dt_sel = dt_sel[keep]

    # времена
    N = x0.size(0)
    t0 = torch.zeros(N, dtype=hist.dtype, device=hist.device)
    t1 = dt_sel

    return x0, x0_class, x1, t0, t1



def make_model_collate(dataset, memory=4, dim=2):
    """
        function for UnifiedDataset, Dataloader from trajData will provide 
        batch to this function. 

        :param memory: define the lenght of agent history provided to model
        :param dim: number of dimensions of X provided to model

        :return: batch with required dividing  
    """
    base_collate = dataset.get_collate_fn()

    def collate_fn(scene_list):
        # scene_list:SceneBatchElement (len = batch_size DataLoader’а)
        agent_batch = base_collate(scene_list)               # -> AgentBatch
        return agentbatch_to_modelpack(agent_batch, memory=memory, dim=dim)
    return collate_fn
