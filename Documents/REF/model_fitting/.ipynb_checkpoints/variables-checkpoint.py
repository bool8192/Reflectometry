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
    [[10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
    [[1.0, 90.0], [0.0, 5.514], [0, 0.01]],
    [[1.0, 90.0], [5.514, 8.2589], [0, 0.01]],
    [[1.0, 90.0], [8.2589, 13.7156], [0, 0.01]],
    [[1.0, 90.0], [13.7156, 19.1769], [0, 0.01]],
    [[1.0, 90.0], [19.1769, 24.6431], [0, 0.01]],
    [[1.0, 90.0], [24.6431, 30.1127], [0, 0.01]],
    [[1.0, 90.0], [30.1127, 35.5863], [0, 0.01]],
    [[1.0, 90.0], [35.5863, 41.0636], [0, 0.01]],
    [[1.0, 90.0], [41.0636, 46.5446], [0, 0.01]],
    [[1.0, 90.0], [46.5446, 52.0292], [0, 0.01]],
    [[1.0, 90.0], [52.0292, 57.5174], [0, 0.01]],
    [[1.0, 90.0], [57.5174, 63.009], [0, 0.01]],
    [[1.0, 90.0], [63.009, 68.5038], [0, 0.01]],
    [[1.0, 90.0], [68.5038, 73.9996], [0, 0.01]],
    [[1.0, 90.0], [73.9996, 77.9919], [0, 0.01]],
    [[1.0, 90.0], [77.9919, 80.00], [0, 0.01]],
    [[500.0, 900.0],[84.2, 92.950], [0, 0.01]],
    [[0.1, 40.0], [80.575, 84.200], [0, 0.01]],
    [[0.1, 40.0], [75.95, 80.5750], [0, 0.01]],
    [[0.1, 40.0], [71.375, 75.950], [0, 0.01]],
    [[0.1, 40.0], [66.700, 71.375], [0, 0.01]],
    [[0.1, 40.0], [62.075, 66.700], [0, 0.01]],
    [[0.1, 40.0], [57.450, 62.075], [0, 0.01]],
    [[0.1, 40.0], [52.825, 57.450], [0, 0.01]],
    [[0.1, 40.0], [48.200, 52.825], [0, 0.01]],
    [[0.1, 40.0], [43.575, 48.200], [0, 0.01]],
    [[0.1, 40.0], [38.950, 43.575], [0, 0.01]],
    [[0.1, 40.0], [34.325, 38.950], [0, 0.01]],
    [[0.1, 40.0], [29.700, 34.325], [0, 0.01]],
    [[0.1, 40.0], [25.075, 29.700], [0, 0.01]],
    [[0.1, 40.0], [20.450, 25.075], [0, 0.01]],
    [[0.1, 40.0], [14.825, 20.450], [0, 0.01]],
    [[0.1, 40.0], [9.2000, 14.825], [0, 0.01]],
    [[99.9, 100.0], [7.00, 9.200], [0.0, 0.01]],

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

I_bounds =  torch.tensor([0.997*I0.clone().detach() , 1.003*I0.clone().detach()]).requires_grad_(True)
Ibkg_bounds = torch.tensor([0.0, 3*Ibkg.clone().detach()]).requires_grad_(True)

sigma_bounds1 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)
sigma_bounds2 =  torch.tensor([0.001 , 0.011]).requires_grad_(False)

