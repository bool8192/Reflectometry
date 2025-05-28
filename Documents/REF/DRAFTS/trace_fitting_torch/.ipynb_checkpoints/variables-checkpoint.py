import torch

Ndots :int = 100
rough_res :int = 16
kmax :int = 1.45e+9
dq :int = 0.2e+8
sigma = torch.tensor(0.01).requires_grad_(True)
sigma1  = torch.tensor(0.1).requires_grad_(True)
sigma2  = torch.tensor(0.01).requires_grad_(True)
Ndots_norm :int = 11
gap :int = 0.1
Ndots_trace :int = 4