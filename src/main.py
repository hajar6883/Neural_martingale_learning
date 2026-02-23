import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from bermudan.neural_martingale import *
import torch 


S_t  = np.array([107.66, 99, 97.02, 115, 120]) # (5,)

t_idx = 1
n_times = 5
feats = make_features(torch.from_numpy(S_t),t_idx, n_times )
# print(feats)
# print(feats.shape)

f_net = MLP(2, 16, 1)
g_net = MLP(2, 16, 1)

import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 4
loss = z**2

print(loss.grad_fn)