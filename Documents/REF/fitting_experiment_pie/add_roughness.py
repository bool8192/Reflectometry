import torch
import math as ma
import plotly.graph_objects as go


def transform_array(input_array, N):
    device = input_array.device
    
    initial_part = torch.cat(((input_array[:, 0] - input_array[:, 2]).unsqueeze(1), input_array[:, 1].unsqueeze(1)), dim=1)
    
    result = [initial_part[i] for i in range(len(initial_part))]
    
    for i in range(1, len(input_array)):
        d2, ro2, rough2 = input_array[i]
        ro1 = input_array[i-1][1]
        insert_index = len(result) - (len(input_array) - i)
        new_d = (rough2) / N
        k = torch.arange(0, N, device=device)
        new_ro = ((torch.erf(-4*k/N+2)+1)/2)*(ro2-ro1)+ro1
        #new_ro = ((-k+N-0.5)/N)*(ro2-ro1)+ro1
        for j in range(0, N):
            result.insert(insert_index, torch.stack([new_d, new_ro[j]]))
            
    final_result = torch.stack(result)
    return final_result


def plot_density_profile(matr):
    matrix = matr.clone()
    depth_bins = torch.cumsum((1.e+0) * matrix[:, 0].real, dim=0)
    depth_bins = torch.cat(
        (depth_bins, (depth_bins[-1] + (1.e+0) * matrix[-1, 0].real).unsqueeze(0)),
        dim=0
    )
    U = (1.e-0) * matrix[:, 1].real
    U = torch.cat((U, U[-1].unsqueeze(0)))

    x_fill = torch.repeat_interleave(depth_bins[:-1], 2)
    y_fill = torch.repeat_interleave(U, 2)

    depth_bins_np = depth_bins.detach().cpu().numpy()
    U_np = U.detach().cpu().numpy()
    x_fill_np = x_fill.detach().cpu().numpy()
    y_fill_np = y_fill.detach().cpu().numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_fill_np,
        y=y_fill_np,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(1,50,32,0.15)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=depth_bins_np[:-1],
        y=U_np,
        mode='lines',
        line=dict(color='darkgreen', shape='hv'),
        showlegend=False
    ))
    fig.add_hline(y=0, line=dict(color='black', width=2))
    fig.update_layout(
        xaxis_title='Глубина, Å',
        yaxis_title='Плотность длины рассеяния, 10^-6 Å^2',
        xaxis=dict(range=[0, float(depth_bins_np[-1])]),
        showlegend=False,
        plot_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig