import torch.optim as optim
from variables import *
from objective_function import *
from tqdm import trange
from tqdm import tqdm
import torch.nn as nn


def twin_plotter(matr, q, r, var_sigma1, var_sigma2, I0, var_Ibkg, var_I):
    r2, r_conv2 = reflectometry(q, matr, var_sigma1, var_sigma2, var_I, var_Ibkg)
    fig, ax = plt.subplots(figsize=(15, 2.6))

    ax.plot(q.cpu().detach().numpy()[0:len(r.tolist())] * 1e-10, (r).tolist())
    ax.plot(q.cpu().detach().numpy()[0:len(r_conv2.tolist())] * 1e-10, (var_I * r_conv2).tolist())

    ax.set_yscale('log')
    plt.grid()
    plt.show()


def adamw(q, r, func,
          sigma1, sigma2,
          initial_d, initial_r_rho, initial_i_rho, initial_I, initial_Ibkg,
          d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
          betas=(0.99, 0.999), gamma=0.9, wd=0,
          max_iter=1000, k=2, lr=1.2, tol=1e-20):
    k -= 1

    x_de = initial_d.reshape(-1, 5)[:, 0:2].flatten().clone().detach().requires_grad_(True)
    x_be = initial_d.reshape(-1, 5)[:, 2:].flatten().clone().detach().requires_grad_(True)
    x_r_rho = initial_r_rho.clone().detach().requires_grad_(True)
    x_i_rho = initial_i_rho.clone().detach().requires_grad_(True)
    x_I = initial_I.clone().detach().requires_grad_(True)
    x_Ibkg = initial_Ibkg.clone().detach().requires_grad_(True)

    x_I.retain_grad()
    x_Ibkg.retain_grad()

    optimizer = optim.AdamW([
        {'params': x_de, 'lr': 2 * lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_be, 'lr': 0.1 * lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_r_rho, 'lr': 1 * lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_i_rho, 'lr': lr/20 , 'weight_decay': wd, 'betas': betas},
        {'params': x_I, 'lr': 50.0 * lr, 'weight_decay': wd, 'betas': (0.4, 0.8)},
        {'params': x_Ibkg, 'lr': 0.4 * lr, 'weight_decay': wd, 'betas': betas}
    ])
    x_d = torch.cat((x_de.reshape(-1, 2), x_be.reshape(-1, 3)), dim=1).flatten()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    prev_loss = None
    rel_losses = []

    def current_loss_fn_wrapper():
        return func(torch.cat((x_de.reshape(-1, 2), x_be.reshape(-1, 3)), dim=1).flatten(), torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg)

    initial_loss_value = func(torch.cat((x_de.reshape(-1, 2), x_be.reshape(-1, 3)), dim=1).flatten(), torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg).item()

    trim_idx = slice(None, -1)  # =[:-1]
    tg_x_rho = (x_i_rho[trim_idx] / x_r_rho[trim_idx]).detach()

    t = tqdm(range(max_iter), desc="Optimizing", ncols=120)
    for i in t:
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                # собираем x_d из частей
                x_d = torch.cat(
                    (x_de.reshape(-1, 2), x_be.reshape(-1, 3)), dim=1
                ).flatten()

                current_loss = func(
                    torch.cat((x_de.reshape(-1, 2), x_be.reshape(-1, 3)), dim=1).flatten(), torch.complex(x_r_rho, x_i_rho),
                    sigma1, sigma2, x_I, x_Ibkg
                )

            current_loss.backward()

            # --- clip только по leaf-параметрам ---
            torch.nn.utils.clip_grad_norm_(x_de, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_be, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_r_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_i_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_I, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_Ibkg, max_norm=1.0)

            # --- ручная коррекция градиентов ---
            with torch.no_grad():
                grad_r_rho_trimmed = x_r_rho.grad[trim_idx]
                grad_i_rho_trimmed = x_i_rho.grad[trim_idx]
                corrected_grad_r_rho = (grad_i_rho_trimmed + grad_r_rho_trimmed) / 2
                x_r_rho.grad[trim_idx] = corrected_grad_r_rho


            optimizer.step()

            # --- clamp по границам ---
            with torch.no_grad():
                x_de.data = torch.clamp(x_de.data, d_bounds.reshape(-1,5, 2)[:,0:2,0].flatten(), d_bounds.reshape(-1,5, 2)[:,0:2,1].flatten())
                x_be.data = torch.clamp(x_be.data, d_bounds.reshape(-1,5, 2)[:,2:,0].flatten(), d_bounds.reshape(-1,5, 2)[:,2:,1].flatten())
                x_r_rho.data = torch.clamp(x_r_rho.data, r_rho_bounds[:, 0], r_rho_bounds[:, 1])
                x_I.data = torch.clamp_(x_I.data, I_bounds[0], I_bounds[1])
                x_Ibkg.data = torch.clamp(x_Ibkg.data, Ibkg_bounds[0], Ibkg_bounds[1])
                x_i_rho.data[trim_idx] = x_r_rho.data[trim_idx] * tg_x_rho

            scheduler.step()
            rel_losses.append((current_loss / initial_loss_value).item())

            # --- прогресс-бар ---
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                t.set_postfix({
                    "lr": f"{current_lr:.2e}"
                })
            #if i % 50 == 0:
            #    print(x_be.grad)

        except RuntimeError as e:
            print(f"Ошибка на итерации {i}: {str(e)}")
            print("Текущий loss:", current_loss)
            raise

        if prev_loss is not None and abs(prev_loss - current_loss.item()) < tol and i >= 50:
            print(f"\nConverged at iteration {i}")
            break

        prev_loss = current_loss.item()

    return x_d.detach(), x_r_rho.detach(), x_i_rho.detach(), x_I.detach(), x_Ibkg.detach(), current_loss.item(), rel_losses



def adamw_freeform(q, r, func,
          sigma1, sigma2,
          initial_d, initial_r_rho, initial_i_rho, initial_I, initial_Ibkg,
          d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
          betas=(0.99, 0.999), gamma=0.9, wd=0,
          max_iter=1000, k=2, lr=1.2, tol=1e-10):
    k -= 1

    x_d = initial_d.clone().detach().requires_grad_(True)
    x_r_rho = initial_r_rho.clone().detach().requires_grad_(True)
    x_i_rho = initial_i_rho.clone().detach().requires_grad_(True)
    x_I = initial_I.clone().detach().requires_grad_(True)
    x_Ibkg = initial_Ibkg.clone().detach().requires_grad_(True)

    x_I.retain_grad()
    x_Ibkg.retain_grad()

    optimizer = optim.AdamW([
        {'params': x_d, 'lr': 2 * lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_r_rho, 'lr': 1 * lr, 'weight_decay': wd, 'betas': betas},
        {'params': x_i_rho, 'lr': lr / 20, 'weight_decay': wd, 'betas': betas},
        {'params': x_I, 'lr': 1.0e+3 * lr, 'weight_decay': wd, 'betas': (0.4, 0.8)},
        {'params': x_Ibkg, 'lr': 2 * lr, 'weight_decay': wd, 'betas': betas}
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    prev_loss = None
    rel_losses = []

    def current_loss_fn_wrapper():
        return func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg)

    initial_loss_value = func(x_d, torch.complex(x_r_rho, x_i_rho), sigma1, sigma2, x_I, x_Ibkg).item()

    trim_idx = slice(None, -1)  # =[:-1]
    tg_x_rho = (x_i_rho[trim_idx]/ (1e-10+x_r_rho[trim_idx])).detach()

    t = tqdm(range(max_iter), desc="Optimizing", ncols=120)
    for i in t:
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                current_loss = func(
                    x_d, torch.complex(x_r_rho, x_i_rho),
                    sigma1, sigma2, x_I, x_Ibkg
                )

            current_loss.backward()

            # --- clip только по leaf-параметрам ---
            torch.nn.utils.clip_grad_norm_(x_d, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_r_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_i_rho, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_I, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(x_Ibkg, max_norm=1.0)

            # --- ручная коррекция градиентов ---
            with torch.no_grad():
                grad_r_rho_trimmed = x_r_rho.grad[trim_idx]
                grad_i_rho_trimmed = x_i_rho.grad[trim_idx]
                corrected_grad_r_rho = (grad_i_rho_trimmed + grad_r_rho_trimmed) / 2
                x_r_rho.grad[trim_idx] = corrected_grad_r_rho
            optimizer.step()

            # --- clamp по границам ---
            #print(d_bounds.reshape(-1,2, 2)[:,0:2,0].shape, x_d.shape)

            with torch.no_grad():
                #x_d.data = torch.clamp(x_d.data, d_bounds.reshape(-1,2, 2)[:,0:2,0], d_bounds.reshape(-1,2, 2)[:,0:2,1])
                #x_r_rho.data = torch.clamp(x_r_rho.data, r_rho_bounds[:, 0], r_rho_bounds[:, 1])
                #x_i_rho.data = torch.clamp(x_i_rho.data, i_rho_bounds[:, 0], i_rho_bounds[:, 1])
                #x_I.data = torch.clamp_(x_I.data, I_bounds[0], I_bounds[1])
                #x_Ibkg.data = torch.clamp(x_Ibkg.data, Ibkg_bounds[0], Ibkg_bounds[1])
                x_i_rho.data[trim_idx] = x_r_rho.data[trim_idx] * tg_x_rho

            scheduler.step()
            rel_losses.append((current_loss / initial_loss_value).item())

            # --- прогресс-бар ---
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                t.set_postfix({
                    "lr": f"{current_lr:.2e}",
                    "loss": f"{current_loss:.2e}",
                })
            #if i % 50 == 0:
            #    print(x_be.grad)

        except RuntimeError as e:
            print(f"Ошибка на итерации {i}: {str(e)}")
            print("Текущий loss:", current_loss)
            raise

        if prev_loss is not None and abs(prev_loss - current_loss.item()) < tol and i >= 50:
            print(f"\nConverged at iteration {i}")
            break

        prev_loss = current_loss.item()

    return x_d.detach(), x_r_rho.detach(), x_i_rho.detach(), x_I.detach(), x_Ibkg.detach(), current_loss.item(), rel_losses



def pie_optimizer(q, r, betas, gamma, wd,
                  initial_d,
                  initial_r_r, initial_r_i,
                  I0, Ibkg,
                  sigma1, sigma2,
                  learning_rates=None,  # Список скоростей обучения
                  iterations=None,     # Список чисел итераций
                  loss_classes=None,     # Список экземпляров классов с методами objective_function
                  model_type='model'
                  ):

    # Готовим начальные значения перед началом первого этапа
    ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg = (
        initial_d.clone(),
        initial_r_r.clone(),
        initial_r_i.clone(),
        I0.clone(),
        Ibkg.clone()
    )

    # Массив для накопления относительных потерь
    rel_losses_total = []

    # Основной цикл по этапам оптимизации
    for idx, (lr, iters, loss_class) in enumerate(zip(learning_rates, iterations, loss_classes)):
        init_temp_loss = loss_class.objective_function(ans_d, torch.complex(ans_r_rho, ans_i_rho), initial_sigma1, initial_sigma2, I0, ans_Ibkg)
        # Пройти оптимизацию с заданным экземпляром класса и его методом objective_function
        if model_type=='model':
            ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss, rel_losses_stage = adamw(q, r, loss_class.objective_function,
                                                      initial_sigma1, initial_sigma2,
                                                      ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg,
                                                      d_bounds, r_rho_bounds, i_rho_bounds, I_bounds, Ibkg_bounds,
                                                      betas=betas, gamma=gamma, wd=wd,
                                                      lr=lr, max_iter=iters, k=3)
        else:
            ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss, rel_losses_stage = adamw_freeform(q, r,
                                                                                         loss_class.objective_function,
                                                                                         initial_sigma1, initial_sigma2,
                                                                                         ans_d, ans_r_rho, ans_i_rho,
                                                                                         ans_I, ans_Ibkg,
                                                                                         d_bounds_ff, r_rho_bounds_ff,
                                                                                         i_rho_bounds_ff, I_bounds,
                                                                                         Ibkg_bounds,
                                                                                         betas=betas, gamma=gamma,
                                                                                         wd=wd,
                                                                                         lr=lr, max_iter=iters, k=3)


        print(f"Stage {idx+1}: Learning Rate={lr}, Iterations={iters}, relative_loss={loss_class.objective_function(ans_d, torch.complex(ans_r_rho, ans_i_rho), initial_sigma1, initial_sigma2, I0, ans_Ibkg)/init_temp_loss}")

        combined_x_for_print = torch.cat((ans_d.reshape(-1, 5), torch.complex(ans_r_rho, ans_i_rho).reshape(-1, 1)), dim=1)[:, torch.tensor([0, 5, 1, 2, 3, 4])]
        matroxx = torch.cat((matr[0, :].reshape(-1, 6), combined_x_for_print.reshape(-1, 6)), axis=0)
        twin_plotter(matroxx, q, r * ans_I, sigma1, sigma2, I0, ans_Ibkg, ans_I)
        # Применяем правило нормализации относительно предыдущих потерь
        if idx > 0:
            rel_losses_stage = list(map(lambda x: x * rel_losses_total[-1], rel_losses_stage))

        # Накапливаем потери текущего этапа
        rel_losses_total.extend(rel_losses_stage)

    # Последняя стадия оптимизации после дифференциальной эволюции
    from scipy.optimize import differential_evolution as de
    loss_function = Comparator(q, r)
    if model_type=='model':
        objective_function_sigma = varsigma(ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, loss_function.compare, all_bounds)

        bounds = [sigma_bounds1, sigma_bounds2]
        ans_sigma1, ans_sigma2 = de(func=objective_function_sigma.objective_function, bounds=bounds).x
        print(f"Stage DE: sigma1= {ans_sigma1}, sigma2= {ans_sigma2}")
        twin_plotter(matroxx, q, r * ans_I, ans_sigma1, ans_sigma2, I0, ans_Ibkg, ans_I)
    else: ans_sigma1, ans_sigma2 = sigma1, sigma2

    return ans_d, ans_r_rho, ans_i_rho, ans_I, ans_Ibkg, ans_sigma1, ans_sigma2, loss, rel_losses_total