import torch
import numpy as np
import math
from read_complex_matrix import read_complex_matrix

Ndots :int = 200
rough_res :int = 32
kmax :int = 1.45e+9
dq :int = 0.2e+8
sigma = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
sigma1  = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
sigma2  = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
Ndots_norm :int = 11
gap :int = 0.1
Ndots_trace :int = 4
deltaq :float = 0.0
Ibkg = torch.nn.Parameter(torch.tensor(1.1, dtype=torch.float64))



data = np.loadtxt('experiment.txt')

r0 = data[0:, 1]
r2 = data[:, 1] + data[:, 2]
r1 = data[:, 1] - data[:, 2]
sigma = sigma+0.0
Ibkg=Ibkg+0.0
I0_plus = np.mean(r0[r0 > 0.9*r1[0]][0:1+math.ceil(len(r0[r0 > 0.9*r1[0]])*0.7)])
I0 = (torch.tensor(I0_plus).requires_grad_(True) - Ibkg)



matr = read_complex_matrix('ref_matrix.txt')
all_bounds = torch.tensor([
    [[10.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[600.0, 1100.0], [84.2, 92.950], [0, 300], [0.0, 0.55], [0.2, 0.8], [0.45, 1.0]],
    [[399.9, 400.0], [7.00, 9.200], [0, 300], [0.0, 0.55], [0.2, 0.8], [0.45, 1.0]]

]).requires_grad_(True)



np_bounds = np.array([[600, 1200],
                      [1,   500],
                      [0,  1.0],
                      [0, 1.0],
                      [0, 1.0],
                      [1,   700],
                      [0,   1.0],
                      [0,  1.0],
                      [0,  1.0]])

valid_bounds = all_bounds[(all_bounds[..., 0] != all_bounds[..., 1])]
mask_d = []
mask_rho = []
for i in range(len(valid_bounds)):
    if i%6 == 1: mask_rho.append(i)
    else: mask_d.append(i)
r_rho_bounds = valid_bounds[mask_rho]
d_bounds = valid_bounds[mask_d]


valid_matr = matr[(all_bounds[..., 0] != all_bounds[..., 1])].reshape(-1,6)
initial_d = valid_matr[:, [0,2,3,4,5]].real.flatten()
initial_r_rho = valid_matr[:, 1].real.flatten()
initial_i_rho = valid_matr[:, 1].imag.flatten()

i_rho_bounds = torch.cat((0.1*initial_i_rho.reshape(-1, 1), 1.9*initial_i_rho.reshape(-1, 1)), axis = 1)

initial_sigma1 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
initial_sigma2 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

I_bounds =  torch.tensor([0.997*I0.clone().detach() , 1.003*I0.clone().detach()]).requires_grad_(True)
Ibkg_bounds = torch.tensor([0.0, 3*Ibkg.clone().detach()]).requires_grad_(True)

sigma_bounds1 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)
sigma_bounds2 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)

