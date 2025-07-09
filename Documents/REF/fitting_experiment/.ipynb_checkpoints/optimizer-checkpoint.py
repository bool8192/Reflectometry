import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from variables import *
from ref import reflectometry

def repair_matrix(bounds, vector):
       mask = bounds[..., 0] != bounds[..., 1]
       flat_mask = mask.flatten()
       indices_flat = flat_mask.nonzero().squeeze()
       flat_base = bounds[..., 0].flatten()
       result = flat_base.scatter(0, indices_flat, vector)
       target_shape = bounds[..., 0].shape
       viewed_result = result.view_as(bounds[..., 0])
       return viewed_result

def twin_plotter(matr, q, r, var_sigma1, var_sigma2, I0, var_Ibkg, var_I):
    r2, r_conv2 = reflectometry(q, matr, var_sigma1, var_sigma2, var_I, var_Ibkg)
    fig, ax = plt.subplots(figsize=(15, 2.6))

    ax.plot(q.cpu().detach().numpy()[0:len(r.tolist())]*1e-9, (I0*r).tolist())
    ax.plot(q.cpu().detach().numpy()[0:len(r_conv2.tolist())]*1e-9, (var_I*r_conv2).tolist())

    ax.set_yscale('log')

    plt.show()


def calculate_numerical_gradient(param_tensor, loss_fn, epsilon=5e-1):
    original_value = param_tensor.item()
    
    param_tensor.data = torch.tensor(original_value + epsilon, device=param_tensor.device, dtype=param_tensor.dtype)
    loss_plus_epsilon = loss_fn()

    param_tensor.data = torch.tensor(original_value - epsilon, device=param_tensor.device, dtype=param_tensor.dtype)
    loss_minus_epsilon = loss_fn()

    param_tensor.data = torch.tensor(original_value, device=param_tensor.device, dtype=param_tensor.dtype)

    grad_numerical = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
    return grad_numerical

    
def adamw(q, r, func, initial_d, initial_rho, initial_sigma1, initial_sigma2, initial_I, initial_Ibkg, d_bounds, rho_bounds, sigma_bounds1, sigma_bounds2, I_bounds, Ibkg_bounds, max_iter, k, lr=1.2, tol=1e-18):
    k -= 1

    x_d = initial_d.clone().detach().requires_grad_(True)
    x_rho = initial_rho.clone().detach().requires_grad_(True)
    x_sigma1 = initial_sigma1.clone().detach().requires_grad_(False)
    x_sigma2 = initial_sigma2.clone().detach().requires_grad_(False)
    x_I = initial_I.clone().detach().requires_grad_(True)
    x_Ibkg = initial_Ibkg.clone().detach().requires_grad_(True)

    x_I.retain_grad()
    x_Ibkg.retain_grad()

    optimizer = optim.AdamW([
        {'params': x_d, 'lr': 1000*lr, 'weight_decay': 0, 'betas': (0.7, 0.98)},
        {'params': x_rho, 'lr': lr, 'weight_decay': 0, 'betas': (0.7, 0.98)},
        {'params': x_I, 'lr': 1.0e+13*lr, 'weight_decay': 0, 'betas': (0.4, 0.8)},
        {'params': x_Ibkg, 'lr': 20*lr, 'weight_decay': 0, 'betas': (0.7, 0.98)}
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.93)
    prev_loss = None
    rel_losses = []

    def current_loss_fn_wrapper():
        return func(x_d, x_rho, x_sigma1, x_sigma2, x_I, x_Ibkg)

    initial_loss_value = func(x_d, x_rho, x_sigma1, x_sigma2, x_I, x_Ibkg).item()

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.enable_grad():
                current_loss = current_loss_fn_wrapper()

            current_loss.backward()

            grad_sigma1_numerical = calculate_numerical_gradient(x_sigma1, current_loss_fn_wrapper, epsilon=8e-4)
            x_sigma1.grad = grad_sigma1_numerical

            grad_sigma2_numerical = calculate_numerical_gradient(x_sigma2, current_loss_fn_wrapper, epsilon=8e-4)
            x_sigma2.grad = grad_sigma2_numerical

            optimizer.step()

            current_lr_sigma = lr
            with torch.no_grad():
                x_sigma1.data -= current_lr_sigma * x_sigma1.grad
                x_sigma2.data -= current_lr_sigma * x_sigma2.grad

            with torch.no_grad():
                x_d.data = torch.clamp(x_d.data, d_bounds[:,0], d_bounds[:,1])
                x_rho.data = torch.clamp(x_rho.data, rho_bounds[:, 0], rho_bounds[:, 1])
                x_sigma1.data = torch.clamp(x_sigma1.data, sigma_bounds1[0], sigma_bounds1[1])
                x_sigma2.data = torch.clamp(x_sigma2.data, sigma_bounds2[0], sigma_bounds2[1])
                x_I.data = torch.clamp_(x_I.data, I_bounds[0], I_bounds[1])
                x_Ibkg.data = torch.clamp(x_Ibkg.data, Ibkg_bounds[0], Ibkg_bounds[1])

            scheduler.step()

            rel_losses.append((current_loss / initial_loss_value).item())

            if i % int((max_iter - 1) / k) == 0:
                combined_x_for_print = torch.cat((x_d.reshape(-1, 2), x_rho.reshape(-1, 1)), dim=1)[:,torch.tensor([0, 2, 1])].flatten()

                print("relative_loss:  ", (current_loss / initial_loss_value).item(), "\n",
                      "d:  ", x_d.detach(), "   ",
                      "grad_d:  ", x_d.grad, "\n",
                      "rho:  ", x_rho.detach(), "   ",
                      "grad_rho:  ", x_rho.grad, "\n",
                      "sigma1:  ", x_sigma1.detach(), "   ",
                      "grad_sigma1:  ", x_sigma1.grad, "\n",
                      "sigma2:  ", x_sigma2.detach(), "   ",
                      "grad_sigma2:  ", x_sigma2.grad, "\n",
                      "I:  ", x_I.detach(), "   ",
                      "grad_I:  ", x_I.grad, "\n",
                      "Ibkg:  ", x_Ibkg.detach(), "   ",
                      "grad_Ibkg:  ", x_Ibkg.grad, "\n",
                      "learning_rate:  ", optimizer.param_groups[0]['lr'],  "\n",
                      "iteration:  ", i + 1,  "\n")
                twin_plotter(repair_matrix(all_bounds, combined_x_for_print), q, r, x_sigma1, x_sigma2, initial_I, x_Ibkg, x_I)

        except RuntimeError as e:
            print(f"Ошибка на итерации {i}: {str(e)}")
            print("Текущее значение x_d:", x_d.detach())
            print("Текущее значение x_rho:", x_rho.detach())
            print("Градиент x_d:", x_d.grad)
            print("Градиент x_rho:", x_rho.grad)
            print("Текущий loss:", current_loss)
            print("Текущий LR:", optimizer.param_groups[0]['lr'])
            print("Текущий I:", x_I.detach())
            print("Текущий LR:",x_Ibkg.detach())
            raise

        if prev_loss is not None and abs(prev_loss - current_loss.item()) < tol and i >= 50:
            print(f"Converged at iteration {i}")
            break

        prev_loss = current_loss.item()

    return x_d.detach(), x_rho.detach(), x_sigma1.detach(), x_sigma2.detach(), x_I.detach(), x_Ibkg.detach(), current_loss.item(), rel_losses