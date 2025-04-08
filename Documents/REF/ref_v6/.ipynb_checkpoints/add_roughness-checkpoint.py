import numpy as np
import math as ma
import plotly.graph_objects as go

def transform_array(input_array, N, function):
    result = np.column_stack((input_array[:, 0]- input_array[:, 2], input_array[:, 1]))[:, :2].tolist()  #(ro, d)
    # Добавил промежуточные точки
    for i in range(1, len(input_array)):
        d2, ro2, rough2 = input_array[i]
        ro1 = input_array[i-1][1]
        insert_index = len(result) - (len(input_array) - i)
        for j in range(0, N+1):
            new_d = (rough2) / N
            new_ro =  ((ma.erf(-4*j/N+2)+1)/2)*(ro2-ro1)+ro1
            result.insert(insert_index, [new_d, new_ro])
    return np.array(result, dtype=complex)


def plot_density_profile(matrix):
    matrix = np.array(matrix)
    depth_bins = np.cumsum(np.append(0, (1e+10)*matrix[:, 0].real))
    depth_bins = np.append(depth_bins, depth_bins[-1] + (1e+10)*matrix[-1, 0].real)
    U = (1e-14)*matrix[:, 1].real
    U = np.append(U, U[-1])
    x_fill = np.repeat(depth_bins[:-1], 2)
    y_fill = np.repeat(U, 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(1,50,32,0.15)',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=depth_bins[:-1],
        y=U,
        mode='lines',
        line=dict(color='darkgreen', shape='hv'),
        showlegend=False
    ))
    fig.add_hline(y=0, line=dict(color='black', width=2))
    fig.update_layout(
        xaxis_title='Глубина, Å',
        yaxis_title='Плотность длины рассеяния, 10^-6 Å^2',
        xaxis=dict(range=[0, depth_bins[-1]]),
        showlegend=False,
        plot_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig
