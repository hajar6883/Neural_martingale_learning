import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple


def make_features(S: torch.Tensor, t: int, n_steps: int, K: float) -> torch.Tensor:
    """
    Encode stock price and time into a normalized 2D feature vector.

    Features:
        log(S/K)  — log-moneyness, 0 when ATM, positive OTM for calls
        tau       — time to maturity as fraction of horizon in [0,1]
                    tau=1 at t=0 (full time left), tau=0 at expiry
    """
    log_moneyness = torch.log(S / K).unsqueeze(1)                              # (N,1)
    tau = torch.full((S.shape[0], 1), (n_steps - 1 - t) / (n_steps - 1),
                     device=S.device)                                           # (N,1)
    return torch.cat([log_moneyness, tau], dim=1)                              # (N,2)


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
def construct_neural_martingale(paths_np: np.ndarray, f_net: nn.Module, g_net: nn.Module, K: float) -> np.ndarray:
    device = next(f_net.parameters()).device
    paths = torch.tensor(paths_np, dtype=torch.float32, device=device)
    N, n_steps = paths.shape

    M = torch.zeros((N, n_steps), device=device)
    for t in range(n_steps - 1):
        feat_t   = make_features(paths[:, t],   t,   n_steps, K)  # (N,2)
        feat_tp1 = make_features(paths[:, t+1], t+1, n_steps, K)  # (N,2)

        f = f_net(torch.cat([feat_t, feat_tp1], dim=1)).squeeze(-1)  # (N,) — f takes (feat_t, feat_{t+1})
        g = g_net(feat_t).squeeze(-1)                                 # (N,) — g takes feat_t only

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

    # features are [log(S/K), tau] -> 2 dims per timestep
    # f sees (feat_t, feat_{t+1})  -> in_dim=4
    # g sees feat_t only           -> in_dim=2
    f_net = MLP(in_dim=4, hidden_dim=64, out_dim=1).to(device)
    g_net = MLP(in_dim=2, hidden_dim=64, out_dim=1).to(device)

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
                feat_t   = make_features(paths[:, t],   t,   Tn, K)  # (B,2)
                feat_tp1 = make_features(paths[:, t+1], t+1, Tn, K)  # (B,2)

                f = f_net(torch.cat([feat_t, feat_tp1], dim=1)).squeeze(1)  # (B,)
                g = g_net(feat_t).squeeze(1)                                 # (B,)

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
