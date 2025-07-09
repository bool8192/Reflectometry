import torch

Ndots :int = 200
rough_res :int = 16
kmax :int = 1.45e+9
dq :int = 0.2e+8
sigma = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
sigma1  = torch.nn.Parameter(torch.tensor(0.14, dtype=torch.float64))
sigma2  = torch.nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
Ndots_norm :int = 11
gap :int = 0.1
Ndots_trace :int = 4
deltaq :float = 0.0
Ibkg = torch.nn.Parameter(torch.tensor(1.1, dtype=torch.float64))


# Определение границ
all_bounds = torch.tensor([
    [[10.00, 10.00], [0.00, 0.00], [0.000, 0.000]],
    [[10.0, 9000.0], [4.01, 4.03],  [1.00, 50.00]],
    [[0.000, 40.0], [0.5, 0.65], [16.00, 28.00]],
    [[100.00, 100.00], [1.58, 1.58], [0.000, 0.000]]
]).requires_grad_(True)

valid_bounds = all_bounds[(all_bounds[..., 0] != all_bounds[..., 1])]
mask_d = []
mask_rho = []
for i in range(len(valid_bounds)):
    if i%3 == 1: mask_rho.append(i)
    else: mask_d.append(i)
rho_bounds = valid_bounds[mask_rho]
d_bounds = valid_bounds[mask_d]
