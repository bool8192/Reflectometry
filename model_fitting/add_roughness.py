import torch
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import torch
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from zmq.backend import first


def linear_interpolate_tensor(tensor1, tensor2, num_steps):
    """
    Выполняет линейную интерполяцию между двумя тензорами одинаковой формы с использованием PyTorch.
    Результат совместим с автоматическим дифференцированием.

    Args:
        tensor1 (torch.Tensor): Первый тензор.
        tensor2 (torch.Tensor): Второй тензор.
        num_steps (int): Количество промежуточных шагов (включая начальный и конечный тензоры).

    Returns:
        torch.Tensor: Тензор, содержащий все промежуточные тензоры.
                      Новая первая размерность соответствует количеству шагов.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Тензоры должны иметь одинаковую форму.")
    if num_steps < 2:
        raise ValueError("Количество шагов должно быть не менее 2 (начальный и конечный тензоры).")

    alpha = torch.linspace(0, 1, num_steps, device=tensor1.device, dtype=tensor1.dtype)
    alpha_reshaped = alpha.view(num_steps, *([1] * len(tensor1.shape)))

    interpolated_tensors = tensor1 * (1 - alpha_reshaped) + tensor2 * alpha_reshaped

    return interpolated_tensors


def comb_transformer(matr):
    """""
    #first_gap = matr[3:5, :].unsqueeze(0).repeat(12, 1, 1).view(-1, 9)
    first_gap = linear_interpolate_tensor(matr[3:5, :], matr[5:7, :], 12).view(-1, 6)

    second_gap = linear_interpolate_tensor(matr[10:12, :], matr[12:14, :], 75).view(-1, 6)

    # Собираем итоговый тензор с помощью torch.cat
    comb = torch.cat([
        matr[0:3, :],  # (3, 9)
        first_gap,  # (24, 9)
        matr[7, :].unsqueeze(0),  # (1, 9)
        matr[8:10, :],  # (2, 9)
        second_gap,
        matr[-1, :].unsqueeze(0)  # (1, 9)
    ], dim=0)
    """""
    gap = linear_interpolate_tensor(matr[3:5, :], matr[5:7, :], 89).view(-1, 6)

    comb = torch.cat([
        matr[0:3, :],
        gap,
        matr[-1, :].unsqueeze(0)
    ], dim=0)
    return comb


def transform_array(input_array_untransformed, N):
    device = input_array_untransformed.device

    input_array = comb_transformer(input_array_untransformed)

    input_array_clone = input_array.clone()

    eps = 1e-12  # защита от деления на ноль

    for i in range(1, input_array.shape[0] - 1):
        if input_array[i, 0].real < input_array[i, 2].real:
            _, ro1, _, _, _, _ = input_array[i - 1].real
            d2, ro2, s2, _, _, _ = input_array[i].real
            _, ro3, s3, _, _, _ = input_array[i + 1].real

            denom = (ro2 - ro1) / (s2 + eps) + (ro2 - ro3) / (s3 + eps)

            if denom.abs() < eps:
                print(f"[WARNING] Small denominator at i={i}: denom={denom}, s2={s2}, s3={s3}")
                continue

            s3x = ((ro1 - ro3) + (d2 + s3) * (ro2 - ro1) / (s2 + eps)) / denom
            s2x = d2 + s3 - s3x

            if abs(s3) < eps:
                print(f"[WARNING] Small s3 at i={i}, s3={s3}")
                continue

            ro2x = ro3 * (1 - s3x / (s3 + eps)) + ro2 * (s3x / (s3 + eps))

            if torch.isnan(s3x) or torch.isnan(s2x) or torch.isnan(ro2x):
                print(f"[NaN DETECTED at i={i}]")
                print(f"ro1={ro1}, ro2={ro2}, ro3={ro3}, d2={d2}, s2={s2}, s3={s3}")
                print(f"s3x={s3x}, s2x={s2x}, ro2x={ro2x}")
                raise ValueError("NaN detected in transform_array")

            # теперь безопасно менять
            row_ip1 = input_array_clone[i + 1].clone()
            row_i = input_array_clone[i].clone()
            row_ip1[2] = s3x
            row_i[2] = s2x
            row_i[1] = ro2x
            row_i[0] = s2x
            input_array_clone[i + 1] = row_ip1
            input_array_clone[i] = row_i

    initial_part = torch.cat(
        ((input_array_clone[:, 0] - input_array_clone[:, 2]).unsqueeze(1), input_array_clone[:, 1].unsqueeze(1)),
        dim=1)

    all_parts = [initial_part[0].unsqueeze(0)]

    for i in range(1, len(input_array_clone)):
        d2, ro2, rough2, v1, v2, v3 = input_array_clone[i]
        ro1 = input_array_clone[i - 1][1]
        new_d = rough2 / N
        u = torch.linspace(0.0, 1.0, N, device=device)

        n_bernstein = len(input_array_clone[0]) - 2
        zero = torch.tensor(0.0, dtype=u.dtype, device=u.device)
        one = torch.tensor(1.0, dtype=u.dtype, device=u.device)

        coeffs = torch.stack([zero, v1, v2, v3, one])

        k_tensor = torch.arange(len(coeffs), dtype=u.dtype, device=u.device)

        n_bernstein_tensor = torch.tensor(n_bernstein, dtype=u.dtype, device=u.device)

        lgamma_n_plus_1 = torch.lgamma(n_bernstein_tensor + 1)
        lgamma_k_plus_1 = torch.lgamma(k_tensor + 1)
        lgamma_n_minus_k_plus_1 = torch.lgamma(n_bernstein_tensor - k_tensor + 1)

        log_binom = lgamma_n_plus_1 - lgamma_k_plus_1 - lgamma_n_minus_k_plus_1
        binom_coeffs = torch.exp(log_binom)

        u_k = u.unsqueeze(0).pow(k_tensor.unsqueeze(1))
        one_minus_u_n_minus_k = (1 - u).unsqueeze(0).pow(n_bernstein_tensor - k_tensor.unsqueeze(1))

        F = torch.sum(coeffs.unsqueeze(1) * binom_coeffs.unsqueeze(1) * u_k * one_minus_u_n_minus_k, dim=0)

        new_ro = F * (ro2 - ro1) + ro1

        inserted_points = torch.stack([new_d.repeat(N), new_ro], dim=1)
        all_parts.append(inserted_points)
        all_parts.append(initial_part[i].unsqueeze(0))

    final_result = torch.cat(all_parts, dim=0)
    return final_result


def plot_density_profile(matr):
    matrix = matr.clone()
    depth_bins = torch.cumsum((1.e+0) * matrix[:, 0].real, dim=0)
    U = (1.e-0) * matrix[:, 1].real

    x_fill = torch.cat((torch.tensor([0.0]), torch.repeat_interleave(depth_bins, 2)[:-1]), axis=0)
    y_fill = torch.repeat_interleave(U, 2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_fill.detach().cpu().numpy(), y=y_fill.detach().cpu().numpy(), mode="lines",
                             line=dict(color="darkgreen"), fill="tozeroy"))

    fig.update_xaxes(title_text='Глубина, Å')
    fig.update_yaxes(title_text='Плотность длины рассеяния, 10⁻⁶ Å⁻²')
    fig.update_layout(plot_bgcolor='white')

    return fig