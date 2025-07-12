import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from variables import *
from ref import reflectometry
from scipy.optimize import differential_evolution as de
from objective_function import *

def twin_plotter(matr, q, r, var_sigma1, var_sigma2, I0, var_Ibkg, var_I):
    r2, r_conv2 = reflectometry(q, matr, var_sigma1, var_sigma2, var_I, var_Ibkg)
    fig, ax = plt.subplots(figsize=(15, 2.6))

    ax.plot(q.cpu().detach().numpy()[0:len(r.tolist())]*1e-9, (I0*r).tolist())
    ax.plot(q.cpu().detach().numpy()[0:len(r_conv2.tolist())]*1e-9, (var_I*r_conv2).tolist())

    ax.set_yscale('log')
    plt.grid()
    plt.show()

    
def adamw(q, r, func,
          sigma1, sigma2, 
          initial_d, initial_r_rho, initial_i_rho, initial_I, initial_Ibkg, d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds, max_iter, k, lr=1.2, tol=1e-18):
    k -= 1

    x_d = initial_d.clone().detach().requires_grad_(True)
    x_r_rho = initial_r_rho.clone().detach().requires_grad_(True)
    x_i_rho = initial_i_rho.clone().detach().requires_grad_(True)
    x_I = initial_I.clone().detach().requires_grad_(True)
    x_Ibkg = initial_Ibkg.clone().detach().requires_grad_(True)

    x_I.retain_grad()
    x_Ibkg.retain_grad()

    optimizer = optim.AdamW([
        {'params': x_d, 'lr': 1000*lr, 'weight_decay': 0, 'betas': (0.7, 0.98)},
        {'params': x_r_rho, 'lr': lr, 'weight_decay': 0, 'betas': (0.7, 0.98)},
        {'params': x_i_rho, 'lr': lr*90, 'weight_decay': 0, 'betas': (0.7, 0.98)},
        {'params': x_I, 'lr': 1.0e+13*lr, 'weight_decay': 0, 'betas': (0.4, 0.8)},
        {'params': x_Ibkg, 'lr': 20*lr, 'weight_decay': 0, 'betas': (0.7, 0.98)}
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.98)
    prev_loss = None
    rel_losses = []

    def current_loss_fn_wrapper():
        return func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg)

    initial_loss_value = func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg).item()

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.enable_grad():
                current_loss = current_loss_fn_wrapper()

            current_loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_d.data = torch.clamp(x_d.data, d_bounds[:,0], d_bounds[:,1])
                x_r_rho.data = torch.clamp(x_r_rho.data, r_rho_bounds[:, 0], r_rho_bounds[:, 1])
                x_i_rho.data = torch.clamp(x_i_rho.data, i_rho_bounds[:, 0], i_rho_bounds[:, 1])
                x_I.data = torch.clamp_(x_I.data, I_bounds[0], I_bounds[1])
                x_Ibkg.data = torch.clamp(x_Ibkg.data, Ibkg_bounds[0], Ibkg_bounds[1])

            scheduler.step()

            rel_losses.append((current_loss / initial_loss_value).item())

            if i % int((max_iter - 1) / k) == 0:
                combined_x_for_print = torch.cat((x_d.reshape(-1, 2), torch.complex(x_r_rho, x_i_rho).reshape(-1, 1)), dim=1)[:,torch.tensor([0, 2, 1])].flatten()

                print("relative_loss:  ", (current_loss / initial_loss_value).item(), "\n",
                      "d:  ", x_d.detach(), "   ",
                      "grad_d:  ", x_d.grad, "\n",
                      "rho:  ", torch.complex(x_r_rho.detach(), x_i_rho.detach()), "   ",
                      "grad_rho:  ", torch.complex(x_r_rho.grad, x_i_rho.grad), "\n",
                      "I:  ", x_I.detach(), "   ",
                      "grad_I:  ", x_I.grad, "\n",
                      "Ibkg:  ", x_Ibkg.detach(), "   ",
                      "grad_Ibkg:  ", x_Ibkg.grad, "\n",
                      "learning_rate:  ", optimizer.param_groups[0]['lr'],  "\n",
                      "iteration:  ", i + 1,  "\n")
                twin_plotter(repair_matrix(all_bounds, combined_x_for_print), q, r, sigma1, sigma2, initial_I, x_Ibkg, x_I)

        except RuntimeError as e:
            print(f"Ошибка на итерации {i}: {str(e)}")
            print("Текущее значение x_d:", x_d.detach())
            print("Текущее значение x_rho:", torch.complex(x_r_rho.detach(), x_i_rho.detach()))
            print("Градиент x_d:", x_d.grad)
            print("Градиент x_rho:", torch.complex(x_r_rho.grad, x_i_rho.grad))
            print("Текущий loss:", current_loss)
            print("Текущий LR:", optimizer.param_groups[0]['lr'])
            print("Текущий I:", x_I.detach())
            print("Текущий LR:",x_Ibkg.detach())
            raise

        if prev_loss is not None and abs(prev_loss - current_loss.item()) < tol and i >= 50:
            print(f"Converged at iteration {i}")
            break

        prev_loss = current_loss.item()

    return x_d.detach(), x_r_rho.detach(),  x_i_rho.detach(), x_I.detach(), x_Ibkg.detach(), current_loss.item(), rel_losses


"""
def pie_optimizer(q, r, matr,
                  initial_sigma1, initial_sigma2, initial_d, initial_rho, I, Ibkg, 
                  d_bounds, rho_bounds, I_bounds, Ibkg_bounds, sigma_bounds):

    loss_function = Comparator(q, matr, 10*sigma, 6*sigma, I, Ibkg)
    objective_function = varbounds(all_bounds, loss_function.compare)
    objective_function_weightless = varbounds(all_bounds, loss_function.compare_weightless)
    
    ans_d, ans_rho, ans_I, ans_Ibkg, loss, rel_losses1 = adamw(q, r, objective_function.objective_function, initial_sigma1, initial_sigma2,
                                                                      initial_d, initial_rho, I, Ibkg, 
                                                                      d_bounds, rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.005, max_iter = 120, k=3)

    objective_function_sigma = varsigma(ans_d, ans_rho, ans_I, ans_Ibkg, loss_function.compare, all_bounds)
    
    ans_sigma1, ans_sigma2 = de(func=objective_function_sigma.objective_function, bounds=bounds).x
    
    ans_d, ans_rho, ans_I, ans_Ibkg, loss, rel_losses2 = adamw(q, r, objective_function_weightless.objective_function, 
                                                                      ans_sigma1, ans_sigma2,
                                                                      ans_d, ans_rho, ans_I, ans_Ibkg, 
                                                                      d_bounds, rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.0008, max_iter = 60, k=2)

    objective_function_weightless_sigma = varsigma(ans_d, ans_rho, ans_I, ans_Ibkg, loss_function.compare_weightless, all_bounds)
    
    ans_sigma1, ans_sigma2 = de(
    func=objective_function_weightless_sigma.objective_function,
    bounds=bounds).x

    ans_d, ans_rho, ans_I, ans_Ibkg, loss, rel_losses3 = adamw(q, r, objective_function_weightless.objective_function, 
                                                                      ans_sigma1, ans_sigma2,
                                                                      ans_d, ans_rho, ans_I, ans_Ibkg, 
                                                                      d_bounds, rho_bounds, I_bounds, Ibkg_bounds,
                                                                      lr= 0.0005, max_iter = 200, k=2)
    
    rel_losses2 = list(map(lambda x: x * rel_losses1[-1],  rel_losses2))
    rel_losses3 = list(map(lambda x: x * rel_losses2[-1],  rel_losses3))
    rel_losses = rel_losses1 + rel_losses2 + rel_losses3

    fig, ax = plt.subplots(figsize =(15, 4))

    ax.plot(range(0,len(rel_losses)), rel_losses, label='AdamW')

    ax.set_yscale('log')
    ax.legend()
    plt.show()

    return ans_d, ans_rho, ans_I, ans_Ibkg, ans_sigma1, ans_sigma2
"""