import torch
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def transform_array(input_array, N):
    device = input_array.device
    
    initial_part = torch.cat(((input_array[:, 0] - input_array[:, 2]).unsqueeze(1), input_array[:, 1].unsqueeze(1)), dim=1)
    
    result = [initial_part[i] for i in range(len(initial_part))]
    
    for i in range(1, len(input_array)):
        d2, ro2, rough2, v1, v2, v3, v4, v5, v6 = input_array[i]

        ro1 = input_array[i-1][1]
        insert_index = len(result) - (len(input_array) - i)
        new_d = rough2 / N
        u = torch.linspace(0.0, 1.0, N, device=device)
        #new_ro = (alpha2*(-k+N)/N+(1-alpha2)*(torch.erf(-4*k/N+2)+1)/2)*(ro2-ro1)+ro1

        n = len(input_array[0])-2
        coeffs = torch.as_tensor([0.0, v1, v2, v3, v4, v5, v6, 1.0], dtype=u.dtype, device=u.device)

        # базис Бернштейна
        F = torch.zeros_like(u)
        for k in range(len(coeffs)):
            binom = math.comb(n, k)
            F = F + coeffs[k] * binom * (u ** k) * ((1 - u) ** (n - k))
        new_ro = F*(ro1-ro2)+ro2

        for j in range(0, N):
            result.insert(insert_index, torch.stack([new_d, new_ro[j]]))
            
    final_result = torch.stack(result)
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