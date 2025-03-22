"""
Модуль для визуализации результатов расчетов.
Содержит функции для построения графиков.
"""

import cmath as m
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def prepare_data(r, q, Ndots):
    """
    Подготовка данных для построения графиков.

    Параметры
    ---------
    r : numpy.ndarray
        Массив коэффициентов отражения
    q : numpy.ndarray
        Массив значений волнового вектора
    Ndots : int
        Количество точек данных

    Возвращает
    ----------
    r_real : list
        Реальные части коэффициентов отражения
    r_img : list
        Мнимые части коэффициентов отражения
    r_abs : list
        Абсолютные значения коэффициентов отражения
    q0_a : list
        Масштабированные значения волнового вектора
    """
    r_real = []
    r_img = []
    r_abs = []
    q0_a = []
    
    for i in range(0, Ndots-1):
        r_real.append(r[i].real)
        r_img.append(r[i].imag)
        r_abs.append(m.polar(r[i])[0])  # Используем cmath для расчета абсолютного значения
        q0_a.append(q[i].real * 1e-10)
    
    return r_real, r_img, r_abs, q0_a


def plot_complex_reflection(r, q, Ndots):
    """
    Построение графика комплексного коэффициента отражения.

    Параметры
    ---------
    r : numpy.ndarray
        Массив коэффициентов отражения
    q : numpy.ndarray
        Массив значений волнового вектора
    """
    r_real, r_img, r_abs, q0_a = prepare_data(r, q, Ndots)
    
    fig = px.scatter(x=r_img, y=r_real, color=q0_a,
                     color_continuous_scale='hot',
                     title='complex reflection ratio')
    
    fig.update_layout(
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(constrain='domain')
    )
    fig.update_traces(marker=dict(size=2))
    fig.show()


def plot_reflection_vs_wavevector(r, q, Ndots):
    """
    Построение графика зависимости коэффициента отражения от волнового вектора.

    Параметры
    ---------
    r : numpy.ndarray
        Массив коэффициентов отражения
    q : numpy.ndarray
        Массив значений волнового вектора
    """
    r_real, r_img, r_abs, q0_a = prepare_data(r, q, Ndots)
    
    fig = px.scatter(x=q0_a, y=r_abs,
                     title='График зависимости коэффициента отражения от волнового вектора',
                     labels={'x': 'Значения исходного волнового вектора, Å^-1',
                             'y': 'Значения коэффициента отражения'})
    
    fig.update_yaxes(type='log')
    fig.update_traces(marker=dict(size=2.2))
    fig.show()


def plot_reflection_vs_wavevector_xn(r_list, q, Ndots, labels=None):
    """
    Построение графика зависимости коэффициента отражения от волнового вектора для нескольких наборов данных.

    Параметры
    ---------
    r_list : list of numpy.ndarray
        Список массивов коэффициентов отражения
    q : numpy.ndarray
        Массив значений волнового вектора
    Ndots : int
        Количество точек данных
    labels : list of str, optional
        Список подписей для каждого набора данных
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly  # Цветовая палитра
    
    for i, r_i in enumerate(r_list):
        r_real, r_img, r_abs, q0_a = prepare_data(r_i, q, Ndots)
        
        # Используем кастомные метки или автоматические
        name = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        
        fig.add_trace(
            go.Scatter(
                x=q0_a,
                y=r_abs,
                mode='markers',
                name=name,
                marker=dict(
                    size=2.2,
                    color=colors[i % len(colors)]  # Циклический выбор цвета
                )
            )
        )

    fig.update_layout(
        title='График зависимости коэффициента отражения от волнового вектора',
        xaxis_title='Значения исходного волнового вектора, Å^-1',
        yaxis_title='Значения коэффициента отражения',
        yaxis_type='log',
        legend_title="Легенда"
    )
    
    fig.show()  