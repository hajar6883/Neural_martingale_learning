import numpy as np
import torch
import torch.nn as nn
from typing import Callable


def make_features(S: torch.Tensor, t: int, n_steps: int, K: float) -> torch.Tensor:
    """
    Encode stock price and time into a normalized 2D feature vector.

    Features:
        log(S/K)  — log-moneyness, 0 when ATM
        tau       — time to maturity fraction in [0,1]: 1 at t=0, 0 at expiry
    """
    log_moneyness = torch.log(S / K).unsqueeze(1)
    tau = torch.full((S.shape[0], 1), (n_steps - 1 - t) / (n_steps - 1), device=S.device)
    return torch.cat([log_moneyness, tau], dim=1)  # (N, 2)


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


@torch.no_grad()
def construct_neural_martingale(
    paths_np: np.ndarray,
    h_net: nn.Module,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Build the martingale process M using the stochastic integral parametrization:

        dM_t = h(feat_t) * Z_t

    where Z_t is the risk-neutral Brownian increment recovered from the path:

        Z_t = ( log(S_{t+1}/S_t) - (r - 0.5*sigma^2)*dt ) / (sigma*sqrt(dt))

    Since h(feat_t) is F_t-measurable and E[Z_t | F_t] = 0,
    M is a martingale by construction for any h.

    Returns:
        M: shape (N, n_steps)
    """
    device = next(h_net.parameters()).device
    paths = torch.tensor(paths_np, dtype=torch.float32, device=device)
    N, n_steps = paths.shape
    dt = T / (n_steps - 1)

    M = torch.zeros((N, n_steps), device=device)
    for t in range(n_steps - 1):
        feat_t = make_features(paths[:, t], t, n_steps, K)              # (N, 2)

        log_return = torch.log(paths[:, t+1] / paths[:, t])             # (N,)
        Z_t = (log_return - (r - 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))  # (N,)

        h = h_net(feat_t).squeeze(1)                                     # (N,)
        M[:, t+1] = M[:, t] + h * Z_t

    return M.detach().cpu().numpy()


def train_neural_martingale(
    train_paths_np: np.ndarray,
    payoff_fct: Callable[[np.ndarray, float], np.ndarray],
    K: float,
    r: float,
    sigma: float,
    T_total: float,
    n_epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """
    Train h_net to minimize the dual upper bound:

        E[ max_t ( disc_t * h_t - M_t ) ]

    where dM_t = h_net(feat_t) * Z_t is a martingale by construction.
    No regularization needed.

    Returns:
        h_net: trained network
    """
    train_paths_all = torch.tensor(train_paths_np, dtype=torch.float32, device=device)
    N, Tn = train_paths_all.shape

    dt = T_total / (Tn - 1)
    disc = torch.exp(-r * dt * torch.arange(Tn, device=device, dtype=torch.float32))

    payoff_np = payoff_fct(train_paths_np, K)
    payoff_all = torch.tensor(payoff_np, dtype=torch.float32, device=device)

    # precompute Z_t for all paths: shape (N, Tn-1)
    log_returns = torch.log(train_paths_all[:, 1:] / train_paths_all[:, :-1])
    Z_all = (log_returns - (r - 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))

    h_net = MLP(in_dim=2, hidden_dim=64, out_dim=1).to(device)
    opt = torch.optim.Adam(h_net.parameters(), lr=lr)
    idx = torch.arange(N, device=device)

    for _ in range(n_epochs):
        perm = idx[torch.randperm(N)]

        for start in range(0, N, batch_size):
            batch  = perm[start:start + batch_size]
            paths  = train_paths_all[batch]  # (B, T)
            payoff = payoff_all[batch]        # (B, T)
            Z      = Z_all[batch]             # (B, T-1)
            B = paths.shape[0]

            M = torch.zeros((B, Tn), device=device)
            for t in range(Tn - 1):
                feat_t = make_features(paths[:, t], t, Tn, K)  # (B, 2)
                h = h_net(feat_t).squeeze(1)                    # (B,)
                M[:, t+1] = M[:, t] + h * Z[:, t]     # martingale by construction

            discounted_payoff = payoff * disc.unsqueeze(0)      # (B, T)
            pathwise_max = torch.max(discounted_payoff - M, dim=1).values  # (B,)

            loss = pathwise_max.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return h_net
