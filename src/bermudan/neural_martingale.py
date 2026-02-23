import torch
import numpy as np
import torch.nn as nn 
from typing import Callable, Tuple

class MLP(nn.Module):
    def __init__(self, in_dim : int , hidden : int = 64 , out_dim : int =1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_dim)
        )

    def forward(self, x):
        return self.net(x)


def make_features(S_t: torch.Tensor, t_idx: int, n_times: int) -> torch.Tensor:
    logS = torch.log(torch.clamp(S_t, min=1e-12)).unsqueeze(1)  # (n_paths,1)
    tau = torch.full((S_t.shape[0], 1), float(t_idx) / (n_times - 1), device=S_t.device)
    return torch.cat([logS, tau], dim=1)  # (n_paths,2)

@torch.no_grad()
def build_martingale_from_nets(
            paths_np: np.ndarray,
            f_net: nn.Module,
            g_net: nn.Module,
        ) -> np.ndarray:
    
    device = next(f_net.parameters()).device
    paths = torch.tensor(paths_np, dtype=torch.float32, device=device)  # (N,T)
    N, T = paths.shape

    M = torch.zeros((N, T), device=device)
    for t in range(T - 1):
        x_t = make_features(paths[:, t], t, T)          # (N,2)
        x_tp1 = make_features(paths[:, t+1], t+1, T)    # (N,2)
        inp_f = torch.cat([x_t, x_tp1], dim=1)          # (N,4)

        f = f_net(inp_f).squeeze(1)                     # (N,)
        g = g_net(x_t).squeeze(1)                       # (N,)
        dM = f - g                                      # (N,)
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
                device: str = "cpu",
            ) -> Tuple[nn.Module, nn.Module]:
    """
    Trains f_net, g_net to minimize the dual objective:
        mean_i max_t (disc_t * payoff_i,t - M_i,t) + lam * mean(dM^2)
    """
    paths_all = torch.tensor(train_paths_np, dtype=torch.float32, device=device)  # (N,T)
    N, Tn = paths_all.shape

    dt = T_total / (Tn - 1)
    disc = torch.exp(-r * dt * torch.arange(Tn, device=device, dtype=torch.float32))  # (T,)

    # payoff as tensor
    payoff_np = payoff_fct(train_paths_np, K)  # (N,T)
    payoff_all = torch.tensor(payoff_np, dtype=torch.float32, device=device)

    f_net = MLP(in_dim=4, hidden=64, out_dim=1).to(device)  # input: x_t(2) + x_{t+1}(2)
    g_net = MLP(in_dim=2, hidden=64, out_dim=1).to(device)  # input: x_t(2)
    opt = torch.optim.Adam(list(f_net.parameters()) + list(g_net.parameters()), lr=lr)

    idx = torch.arange(N, device=device)

    for _ in range(n_epochs):
        perm = idx[torch.randperm(N)] # random shuffle for improved training ..
        for start in range(0, N, batch_size):
            batch = perm[start:start+batch_size]
            paths = paths_all[batch]      # (B,T)
            payoff = payoff_all[batch]    # (B,T)
            B = paths.shape[0]

            # build martingale along time for batch
            M = torch.zeros((B, Tn), device=device)
            dM_sq_accum = 0.0   #∑(ΔM)^2 

            for t in range(Tn - 1):
                x_t = make_features(paths[:, t], t, Tn)          # (B,2)
                x_tp1 = make_features(paths[:, t+1], t+1, Tn)    # (B,2)
                inp_f = torch.cat([x_t, x_tp1], dim=1)           # (B,4)

                f = f_net(inp_f).squeeze(1)                      # (B,)
                g = g_net(x_t).squeeze(1)                        # (B,)
                dM = f - g
                dM = dM - dM.mean() # batch centering ( to enforce martingale property E = 0)


                M[:, t+1] = M[:, t] + dM
                dM_sq_accum = dM_sq_accum + (dM * dM).mean()

            discounted_payoff = payoff * disc.unsqueeze(0)       # (B,T)
            dual = discounted_payoff - M                         # (B,T)
            pathwise_max = torch.max(dual, dim=1).values         # (B,)

            loss = pathwise_max.mean() + lam * dM_sq_accum / (Tn - 1)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return f_net, g_net