import torch.optim as optim
from variables import *
from objective_function import *
from tqdm import trange
from tqdm import tqdm

def twin_plotter(matr, q, r, var_sigma1, var_sigma2, I0, var_Ibkg, var_I):
    r2, r_conv2 = reflectometry(q, matr, var_sigma1, var_sigma2, var_I, var_Ibkg)
    fig, ax = plt.subplots(figsize=(15, 2.6))
    
    ax.plot(q.cpu().detach().numpy()[0:len(r.tolist())]*1e-10, (r).tolist())
    ax.plot(q.cpu().detach().numpy()[0:len(r_conv2.tolist())]*1e-10, (var_I*r_conv2).tolist())

    ax.set_yscale('log')
    plt.grid()
    plt.show()

    
def adamw(q, r, func,
          sigma1, sigma2,
          initial_d, initial_r_rho, initial_i_rho, initial_I, initial_Ibkg,
          d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
          betas = (0.99, 0.999), gamma = 0.9, wd = 0,
          max_iter=1000, k=2, lr=1.2, tol=1e-20):
    k -= 1

    x_d = initial_d.clone().detach().requires_grad_(True)
    x_r_rho = initial_r_rho.clone().detach().requires_grad_(True)
    x_i_rho = initial_i_rho.clone().detach().requires_grad_(True)
    x_I = initial_I.clone().detach().requires_grad_(True)
    x_Ibkg = initial_Ibkg.clone().detach().requires_grad_(True)

    x_I.retain_grad()
    x_Ibkg.retain_grad()

    optimizer = optim.AdamW([
        {'params': x_d, 'lr': 2*lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_r_rho, 'lr': 1*lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_i_rho, 'lr': lr/20, 'weight_decay': wd, 'betas': betas},
        {'params': x_I, 'lr': 1.0e+3*lr, 'weight_decay': wd, 'betas': (0.4, 0.8)},
        {'params': x_Ibkg, 'lr': 2*lr, 'weight_decay': wd, 'betas': betas}
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    prev_loss = None
    rel_losses = []

    def current_loss_fn_wrapper():
        return func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg)

    initial_loss_value = func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg).item()

    trim_idx = slice(None, -1)  # =[:-1]
    tg_x_rho = (x_i_rho[trim_idx] / x_r_rho[trim_idx]).detach()

    t = tqdm(range(max_iter), desc="Optimizing", ncols=120)
    for i in t:
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                current_loss = current_loss_fn_wrapper()

            current_loss.backward()
            torch.nn.utils.clip_grad_norm_(x_d, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_r_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_i_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_I, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_Ibkg, max_norm=1.0)

            # ручная коррекция градиентов
            with torch.no_grad():
                grad_r_rho_trimmed = x_r_rho.grad[trim_idx]
                grad_i_rho_trimmed = x_i_rho.grad[trim_idx]
                corrected_grad_r_rho = (grad_i_rho_trimmed + grad_r_rho_trimmed) / 2
                x_r_rho.grad[trim_idx] = corrected_grad_r_rho

            optimizer.step()

            with torch.no_grad():
                x_d.data = torch.clamp(x_d.data, d_bounds[:, 0], d_bounds[:, 1])
                x_r_rho.data = torch.clamp(x_r_rho.data, r_rho_bounds[:, 0], r_rho_bounds[:, 1])
                x_i_rho.data = torch.clamp(x_i_rho.data, i_rho_bounds[:, 0], i_rho_bounds[:, 1])
                x_I.data = torch.clamp_(x_I.data, I_bounds[0], I_bounds[1])
                x_Ibkg.data = torch.clamp(x_Ibkg.data, Ibkg_bounds[0], Ibkg_bounds[1])
                x_i_rho.data[trim_idx] = x_r_rho.data[trim_idx] * tg_x_rho

            scheduler.step()
            rel_losses.append((current_loss / initial_loss_value).item())

            # обновляем прогресс-бар с инфой
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                t.set_postfix({
                #    "loss": f"{current_loss.item():.4e}",
                #    "rel_loss": f"{(current_loss / initial_loss_value).item():.4e}",
                    "lr": f"{current_lr:.2e}"
                })

        except RuntimeError as e:
            print(f"Ошибка на итерации {i}: {str(e)}")
            print("Текущее значение x_d:", x_d.detach())
            print("Текущее значение x_rho:", torch.complex(x_r_rho.detach(), x_i_rho.detach()))
            print("Градиент x_d:", x_d.grad)
            print("Градиент x_rho:", torch.complex(x_r_rho.grad, x_i_rho.grad))
            print("Текущий loss:", current_loss)
            print("Текущий LR:", optimizer.param_groups[0]['lr'])
            print("Текущий I:", x_I.detach())
            print("Текущий Ibkg:", x_Ibkg.detach())
            raise

        if prev_loss is not None and abs(prev_loss - current_loss.item()) < tol and i >= 50:
            print(f"\nConverged at iteration {i}")
            break

        prev_loss = current_loss.item()

    return x_d.detach(), x_r_rho.detach(),  x_i_rho.detach(), x_I.detach(), x_Ibkg.detach(), current_loss.item(), rel_losses
