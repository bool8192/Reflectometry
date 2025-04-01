"""
Модуль для визуализации результатов расчетов.
Содержит функции для построения графиков.
"""

import cmath as m
import plotly.express as px


def plot_complex_reflection(r, dk0, Ndots):
    """
    Построение графика комплексного коэффициента отражения.

    Параметры
    ---------
    r : numpy.ndarray
        Массив коэффициентов отражения
    dk0 : float
        Шаг волнового вектора
    Ndots : int
        Количество точек
    """
    r_real = []
    r_img = []
    q0_a = []
    
    for i in range(0, Ndots):
        r_real.append(r[i].real)
        r_img.append(r[i].imag)
        q0_a.append((i+0.9)*dk0.real*1e-10)
    
    fig = px.scatter(x=r_img, y=r_real, color=q0_a,
                     color_continuous_scale='hot',
                     title='complex reflection ratio')
    
    fig.update_layout(
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(constrain='domain')
    )
    fig.update_traces(marker=dict(size=2))
    fig.show()


def plot_reflection_vs_wavevector(r, dk0, Ndots):
    """
    Построение графика зависимости коэффициента отражения от волнового вектора.

    Параметры
    ---------
    r : numpy.ndarray
        Массив коэффициентов отражения
    dk0 : float
        Шаг волнового вектора
    Ndots : int
        Количество точек
    """
    r_abs = []
    q0_a = []
    
    for i in range(0, Ndots):
        r_abs.append(m.polar(r[i])[0]**2)
        q0_a.append(i*dk0.real*1e-10)
    
    fig = px.scatter(x=q0_a, y=r_abs,
                     title='График зависимости коэффициента отражения от волнового вектора',
                     labels={'x': 'Значения исходного волнового вектора, Å^-1',
                             'y': 'Значения коэффициента отражения'})
    
    fig.update_yaxes(type='log')
    fig.update_traces(marker=dict(size=1.5))
    fig.show()