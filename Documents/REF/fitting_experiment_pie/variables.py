import torch
import numpy as np
import math
from read_complex_matrix import read_complex_matrix

Ndots :int = 200
rough_res :int = 16
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

r = data[0:, 1]
r2 = data[:, 1] + data[:, 2]
r1 = data[:, 1] - data[:, 2]
sigma = sigma+0.0
Ibkg=Ibkg+0.0
I0_plus = np.mean(r[r > 0.9*r1[0]][0:1+math.ceil(len(r[r > 0.9*r1[0]])*0.7)])
I0 = (torch.tensor(I0_plus).requires_grad_(True) - Ibkg)



matr = read_complex_matrix('ref_matrix.txt')
all_bounds = torch.tensor([
    [[10.00, 10.00], [0.00, 0.00], [0.000, 0.000]],
    [[1.0, 90.0], [5.5359, 5.5369], [0, 0.01]],
    [[1.0, 90.0], [16.6078, 16.6088], [0, 0.01]],
    [[1.0, 90.0], [27.6797, 27.6807], [0, 0.01]],
    [[1.0, 90.0], [38.7516, 38.7526], [0, 0.01]],
    [[1.0, 90.0], [49.8234, 49.8244], [0, 0.01]],
    [[1.0, 90.0], [60.8953, 60.8963], [0, 0.01]],
    [[1.0, 90.0], [71.9672, 71.9682], [0, 0.01]],
    [[1.0, 90.0], [83.0391, 83.0401], [0, 0.01]],
    [[500.0, 900.0], [80.00, 97.15], [0.00, 350.00]],
    [[0.1, 8.9], [83.5078, 83.5088], [0, 0.01]],
    [[0.1, 8.9], [73.3734, 73.3744], [0, 0.01]],
    [[0.1, 8.9], [63.2391, 63.2401], [0, 0.01]],
    [[0.1, 8.9], [53.1047, 53.1057], [0, 0.01]],
    [[0.1, 8.9], [42.9703, 42.9713], [0, 0.01]],
    [[0.1, 8.9], [32.8359, 32.8369], [0, 0.01]],
    [[0.1, 8.9], [22.7016, 22.7026], [0, 0.01]],
    [[0.1, 8.9], [12.5672, 12.5682], [0, 0.01]],
    [[1.0, 40.0], [7.0, 97.15], [0.00, 0.01]],
    [[1.0, 9.0], [7.5672, 7.5682], [0, 0.01]],
    [[1.0, 9.0], [7.7016, 7.7026], [0, 0.01]],
    [[1.0, 9.0], [7.8359, 7.8369], [0, 0.01]],
    [[1.0, 9.0], [7.9703, 7.9713], [0, 0.01]],
    [[1.0, 9.0], [8.1047, 8.1057], [0, 0.01]],
    [[1.0, 9.0], [8.2391, 8.2401], [0, 0.01]],
    [[1.0, 9.0], [8.3734, 8.3744], [0, 0.01]],
    [[1.0, 9.0], [8.5078, 8.5088], [0, 0.01]],
    [[99.99, 100.0], [7.00, 9.20], [0.0, 0.01]]
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

i_rho_bounds = torch.cat((0.1*initial_i_rho.reshape(-1, 1), 1.9*initial_i_rho.reshape(-1, 1)), axis = 1)

initial_sigma1 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
initial_sigma2 = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

I_bounds =  torch.tensor([0.997*I0 , 1.003*I0]).requires_grad_(True)
Ibkg_bounds = torch.tensor([0.0, 3*Ibkg]).requires_grad_(True)

sigma_bounds1 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)
sigma_bounds2 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)

