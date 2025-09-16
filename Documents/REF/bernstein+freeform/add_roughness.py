import torch
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import torch
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def transform_array(input_array, N):
    device = input_array.device

    initial_part = torch.cat(((input_array[:, 0] - input_array[:, 2]).unsqueeze(1), input_array[:, 1].unsqueeze(1)),
                             dim=1)

    all_parts = [initial_part[0].unsqueeze(0)]

    for i in range(1, len(input_array)):
        d2, ro2, rough2, v1, v2, v3, v4, v5, v6 = input_array[i]
        ro1 = input_array[i - 1][1]
        new_d = rough2 / N
        u = torch.linspace(0.0, 1.0, N, device=device)

        n_bernstein = len(input_array[0]) - 2
        zero = torch.tensor(0.0, dtype=u.dtype, device=u.device)
        one = torch.tensor(1.0, dtype=u.dtype, device=u.device)

        coeffs = torch.stack([zero, v1, v2, v3, v4, v5, v6, one])

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