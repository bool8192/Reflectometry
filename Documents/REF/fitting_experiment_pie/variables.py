import torch
import numpy as np
import math
from read_complex_matrix import read_complex_matrix

Ndots :int = 200
rough_res :int = 16
kmax :int = 1.45e+9
dq :int = 0.2e+8
Ndots_norm :int = 11
gap :int = 0.1
Ndots_trace :int = 4
deltaq :float = 0.0
Ibkg = torch.nn.Parameter(torch.tensor(1.1, dtype=torch.float64))


data = np.loadtxt('exp.txt')

r = data[0:, 1]
delta_r = 12*np.sqrt(data[0:, 1])
delta_r = data[0:, 2]
Ibkg=Ibkg+0.0
I0_plus = np.mean(r[r > 0.9*r[0]][0:1+math.ceil(len(r[r > 0.9*r[0]])*0.7)])
I0 = (torch.tensor(I0_plus).requires_grad_(True) - Ibkg)



matr = read_complex_matrix('ref_matrix.txt')
all_bounds = torch.tensor([
    [[100.00, 100.00], [0.00, 0.00], [0.000, 0.000]],
    [[000.0, 200.0], [20.00, 97.15], [00.00, 20.00]],
    [[500.0, 900.0], [80.00, 97.15], [00.00, 350.00]],
    [[0.000, 40.0], [7.0, 97.15], [0.00, 20.00]],
    [[99.99, 100.0], [7.00, 9.20], [0.0, 99.99]]
]).requires_grad_(True)

valid_bounds = all_bounds[(all_bounds[..., 0] != all_bounds[..., 1])]
mask_d = []
mask_rho = []
for i in range(len(valid_bounds)):
    if i%3 == 1: mask_rho.append(i)
    else: mask_d.append(i)
r_rho_bounds = valid_bounds[mask_rho]
d_bounds = valid_bounds[mask_d]

valid_matr = matr[(all_bounds[..., 0] != all_bounds[..., 1])].reshape(-1,3)
initial_d = valid_matr[:, [0,2]].real.flatten()
initial_r_rho = valid_matr[:, 1,].real.flatten()
initial_i_rho = valid_matr[:, 1,].imag.flatten()
initial_delta_q = torch.tensor(1e+2, requires_grad = True)

i_rho_bounds = torch.cat((0.1*initial_i_rho.reshape(-1, 1), 1.9*initial_i_rho.reshape(-1, 1)), axis = 1)

initial_sigma1 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
initial_sigma2 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

I_bounds =  torch.tensor([0.97*I0 , 1.03*I0]).requires_grad_(True)
Ibkg_bounds = torch.tensor([0.0, 3*Ibkg]).requires_grad_(True)

sigma_bounds1 =  torch.tensor([0.01 , 0.01]).requires_grad_(False)
sigma_bounds2 =  torch.tensor([0.01 , 0.01]).requires_grad_(False)

delta_q_bounds = torch.tensor([-4.0e+7, 4.0e+7]).requires_grad_(True)

