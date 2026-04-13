import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple

from bermudan.neural_martingale import MLP, make_features


@torch.no_grad()
def construct_neural_martingale_modelfree(
    paths_np: np.ndarray,
    f_net: nn.Module,
    g_net: nn.Module,
    K: float,
) -> np.ndarray:
    """
    Build the martingale process M using the (f, g) parametrization:

        dM_t = f(feat_t, feat_{t+1}) - g(feat_t)

    where g approximates E[f | S_t] via a regression loss during training.

    Model-free: works on any set of simulated or historical paths since it
    does not require knowledge of the underlying dynamics to recover Z_t.

    Limitation: martingale property is only approximate — enforced softly
    via the regression loss. Training can be unstable if the dual loss
    pushes f faster than g can track it.

    Returns:
        M: shape (N, n_steps)
    """
    device = next(f_net.parameters()).device
    paths = torch.tensor(paths_np, dtype=torch.float32, device=device)
    N, n_steps = paths.shape

    M = torch.zeros((N, n_steps), device=device)
    for t in range(n_steps - 1):
        feat_t   = make_features(paths[:, t],   t,   n_steps, K)  # (N, 2)
        feat_tp1 = make_features(paths[:, t+1], t+1, n_steps, K)  # (N, 2)

        f = f_net(torch.cat([feat_t, feat_tp1], dim=1)).squeeze(-1)  # (N,)
        g = g_net(feat_t).squeeze(-1)                                 # (N,)

        M[:, t+1] = M[:, t] + (f - g)

    return M.detach().cpu().numpy()


def train_neural_martingale_modelfree(
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
    """
    Train (f_net, g_net) to minimize the dual upper bound:

        E[ max_t ( disc_t * h_t - M_t ) ]

    where dM_t = f(feat_t, feat_{t+1}) - g(feat_t).

    The martingale property E[dM_t | F_t] = 0 is enforced softly via:
        - lam * E[dM^2]:        variance penalty, prevents M from exploding
        - beta * E[(g - f)^2]:  regression loss, trains g toward E[f | S_t]

    Note: f peeks at S_{t+1} (future information), so the martingale property
    is not guaranteed by construction — it relies on g converging to E[f | S_t].
    Training can be unstable: the dual loss pushes f upward faster than the
    regression loss pulls g along, causing M to drift and the bound to become
    invalid. Requires careful hyperparameter tuning.

    Returns:
        (f_net, g_net): trained networks
    """
    train_paths_all = torch.tensor(train_paths_np, dtype=torch.float32, device=device)
    N, Tn = train_paths_all.shape

    dt = T_total / (Tn - 1)
    disc = torch.exp(-r * dt * torch.arange(Tn, device=device, dtype=torch.float32))

    payoff_np = payoff_fct(train_paths_np, K)
    payoff_all = torch.tensor(payoff_np, dtype=torch.float32, device=device)

    # f sees (feat_t, feat_{t+1}) -> in_dim=4
    # g sees feat_t only          -> in_dim=2
    f_net = MLP(in_dim=4, hidden_dim=64, out_dim=1).to(device)
    g_net = MLP(in_dim=2, hidden_dim=64, out_dim=1).to(device)

    opt = torch.optim.Adam(list(f_net.parameters()) + list(g_net.parameters()), lr=lr)
    idx = torch.arange(N, device=device)

    for _ in range(n_epochs):
        perm = idx[torch.randperm(N)]

        for start in range(0, N, batch_size):
            batch  = perm[start:start + batch_size]
            paths  = train_paths_all[batch]  # (B, T)
            payoff = payoff_all[batch]        # (B, T)
            B = paths.shape[0]

            M = torch.zeros((B, Tn), device=device)
            dM_sq_accum = 0.0
            reg_accum   = 0.0

            for t in range(Tn - 1):
                feat_t   = make_features(paths[:, t],   t,   Tn, K)  # (B, 2)
                feat_tp1 = make_features(paths[:, t+1], t+1, Tn, K)  # (B, 2)

                f = f_net(torch.cat([feat_t, feat_tp1], dim=1)).squeeze(1)  # (B,)
                g = g_net(feat_t).squeeze(1)                                 # (B,)

                dM = f - g
                M[:, t+1] = M[:, t] + dM
                dM_sq_accum = dM_sq_accum + (dM * dM).mean()
                reg_accum   = reg_accum   + ((g - f.detach()) ** 2).mean()

            discounted_payoff = payoff * disc.unsqueeze(0)
            pathwise_max = torch.max(discounted_payoff - M, dim=1).values

            loss = (pathwise_max.mean()
                    + lam  * dM_sq_accum / (Tn - 1)
                    + beta * reg_accum   / (Tn - 1))

            opt.zero_grad()
            loss.backward()
            opt.step()

    return f_net, g_net
