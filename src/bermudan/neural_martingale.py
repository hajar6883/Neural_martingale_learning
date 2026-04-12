import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()  # no compute graph needed during martingale construction
def construct_neural_martingale(paths_np: np.ndarray, f_net: nn.Module, g_net: nn.Module) -> np.ndarray:
    device = next(f_net.parameters()).device
    paths = torch.tensor(paths_np, dtype=torch.float32, device=device)
    N, n_steps = paths.shape

    M = torch.zeros((N, n_steps), device=device)
    for t in range(n_steps - 1):  
        X_t   = paths[:, t].unsqueeze(1)      # (N,1) 
        X_tp1 = paths[:, t+1].unsqueeze(1)    # (N,1)
        f_input = torch.cat([X_t, X_tp1], dim=1)  # (N,2)

        f = f_net(f_input).squeeze(-1)   # (N,)
        g = g_net(X_t).squeeze(-1)       # (N,) 

        dM = f - g
        M[:, t+1] = M[:, t] + dM

    return M.detach().cpu().numpy()


def train_neural_martingale(
    train_paths_np: np.ndarray,
    payoff_fct: Callable[[np.ndarray, float], np.ndarray],
    K: float,
    r: float,
    T_total: float,
    n_epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    lam: float = 1e-3,
    beta: float = 1.0,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module]:

    train_paths_all = torch.tensor(train_paths_np, dtype=torch.float32, device=device)  # (N,T)
    N, Tn = train_paths_all.shape

    dt = T_total / (Tn - 1)
    disc = torch.exp(-r * dt * torch.arange(Tn, device=device, dtype=torch.float32))  # (T,)

    payoff_np = payoff_fct(train_paths_np, K)  # (N,T)
    payoff_all = torch.tensor(payoff_np, dtype=torch.float32, device=device)

    # f sees the transition (S_t, S_{t+1}) -> in_dim=2
    # g sees only current state (S_t,)     -> in_dim=1  
    f_net = MLP(in_dim=2, hidden_dim=64, out_dim=1).to(device)  
    g_net = MLP(in_dim=1, hidden_dim=64, out_dim=1).to(device)

    opt = torch.optim.Adam(list(f_net.parameters()) + list(g_net.parameters()), lr=lr)
    idx = torch.arange(N, device=device)

    for _ in range(n_epochs):
        perm = idx[torch.randperm(N)]

        for start in range(0, N, batch_size):
            batch = perm[start:start + batch_size]
            paths  = train_paths_all[batch]  # (B,T)
            payoff = payoff_all[batch]        # (B,T)
            B = paths.shape[0]

            M = torch.zeros((B, Tn), device=device)
            dM_sq_accum = 0.0
            reg_accum   = 0.0

            for t in range(Tn - 1):
                X_t   = paths[:, t].unsqueeze(1)      # (B,1) 
                X_tp1 = paths[:, t+1].unsqueeze(1)    # (B,1) 
                f_input = torch.cat([X_t, X_tp1], dim=1)  # (B,2)

                f = f_net(f_input).squeeze(1)  # (B,)
                g = g_net(X_t).squeeze(1)      # (B,)

                dM = f - g
                M[:, t+1] = M[:, t] + dM
                dM_sq_accum = dM_sq_accum + (dM * dM).mean()         
                reg_accum   = reg_accum   + ((g - f.detach()) ** 2).mean()  # bug 12: enforce E[dM|S_t]=0

            discounted_payoff = payoff * disc.unsqueeze(0)  # (B,T) 
            dual = discounted_payoff - M
            pathwise_max = torch.max(dual, dim=1).values    # (B,)

            loss = (pathwise_max.mean()
                    + lam  * dM_sq_accum / (Tn - 1)
                    + beta * reg_accum   / (Tn - 1))

            opt.zero_grad()
            loss.backward()
            opt.step() 

    return f_net, g_net
