import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from variables import *
from ref import *
from read_complex_matrix import read_complex_matrix
from calculating_dynamic_resolution import *
torch.set_printoptions(precision=4)

import timeit

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

q = torch.pi*4.0e+10*torch.sin(torch.tensor(np.deg2rad(data[:, 0])))/1.524
r_mod, r_mod_conv= reflectometry(q, matr, sigma, sigma, I0, Ibkg)

fig, ax = plt.subplots(figsize=(18, 8))

ax.fill_between(q*1e-10, r1, r2, color='purple', alpha=0.7, label= 'эксперимент')
ax.plot(q.cpu().detach().numpy()[0:len(r_mod_conv.tolist())]*1e-10, (I0*r_mod_conv).tolist())
ax.set_yscale('log')

ax.legend()
plt.grid()
plt.show()

r = torch.tensor(r/I0.cpu().detach().numpy())

from objective_function import *

loss_function = Comparator(q, r)

objective_function = varbounds(all_bounds, loss_function.compare)
objective_function_weightless = varbounds(all_bounds, loss_function.compare_weightless)

from optimizer import *
ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss, rel_losses1 = adamw(q, r, objective_function.objective_function, initial_sigma1, initial_sigma2,
                                                                      initial_d, initial_r_rho, initial_i_rho, I0, Ibkg, 
                                                                      d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.11, max_iter = 150, k=8)

objective_function_sigma = varsigma(ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss_function.compare, all_bounds)

from scipy.optimize import differential_evolution as de

bounds = [sigma_bounds1,
          sigma_bounds2]

ans_sigma1, ans_sigma2 = de(
    func=objective_function_sigma.objective_function,
    bounds=bounds
).x

ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss, rel_losses2 = adamw(q, r, objective_function_weightless.objective_function, 
                                                                      ans_sigma1, ans_sigma2, 
                                                                      ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, 
                                                                      d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.006, max_iter = 660, k=5)

objective_function_weightless_sigma = varsigma(ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss_function.compare_weightless, all_bounds)
ans_sigma1, ans_sigma2 = de(
    func=objective_function_weightless_sigma.objective_function,
    bounds=bounds,
).x

ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss, rel_losses3 = adamw(q, r, objective_function_weightless.objective_function, 
                                                                      ans_sigma1, ans_sigma2, 
                                                                      ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, 
                                                                      d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.003, max_iter = 2900, k=12)

rel_losses2 = list(map(lambda x: x * rel_losses1[-1],  rel_losses2))
rel_losses3 = list(map(lambda x: x * rel_losses2[-1],  rel_losses3))
rel_losses = rel_losses1 + rel_losses2 + rel_losses3

fig, ax = plt.subplots(figsize =(15, 4))

ax.plot(range(0,len(rel_losses)), rel_losses, label='AdamW')

ax.set_yscale('log')
ax.legend()
plt.grid()
plt.show()

ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss

ans_sigma1, ans_sigma2


