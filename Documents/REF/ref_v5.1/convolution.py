from scipy.stats import norm
import numpy as np
from calculating_dynamic_resolution import *
from variables import *

def convolution(r, qmax, q0, Ndots_norm, sigma01, sigma02, mode):
    """
    Применяет гауссову свертку к массиву комплексных чисел с динамической шириной гауссиана
    
    Parameters:
    r (np.array): входной массив комплексных коэффициентов отражения
    qmax (float): максимальное значение q для нормировки
    q0 (np.array): массив значений q
    Ndots_norm (int): количество точек для построения нормального распределения
    sigma01 (float): ширина гауссиана для свертки в начале диапазона 
    sigma02 (float): ширина гауссиана для свертки в конце диапазона 
    
    Returns:
    np.array: преобразованный массив комплексных чисел
    """
    Ndots = len(r)
    
    # Создаём матрицу сигм для всех точек
    if mode == 'step':
        sigma_matrix = np.array([resolution_function_step(qmax, q, sigma01, sigma02) for q in q0])[:, np.newaxis]
    elif mode == 'smooth_step':
        sigma_matrix = np.array([resolution_function_smooth_step(qmax, q, sigma01, sigma02) for q in q0])[:, np.newaxis]
    else:
        sigma_matrix = np.array([resolution_function(qmax, q, sigma01, sigma02) for q in q0])[:, np.newaxis]
    
    # Создаём веса для окрестных точек
    q_window = np.linspace(1-3*np.max(sigma_matrix), 1+3*np.max(sigma_matrix), Ndots_norm)
    
    # Создаём матрицу весов
    weights_matrix = norm.pdf(q_window, 1, sigma_matrix)
    weights_matrix = weights_matrix / np.sum(weights_matrix, axis=1)[:, np.newaxis]
    
    # Создаём матрицу индексов для всех точек
    q_matrix = q0[:, np.newaxis] * q_window
    indices_matrix = np.searchsorted(q0, q_matrix)
    
    # Маска для индексов, выходящих за границы
    mask = indices_matrix < Ndots
    indices_matrix = indices_matrix * mask
    
    # Создаём маску весов соответствующего размера
    weights_matrix_masked = weights_matrix * mask
    
    # Получаем значения r для всех индексов
    r_matrix = r[indices_matrix] * mask
    
    # Считаем взвешенное среднее
    return np.sum(r_matrix * weights_matrix_masked, axis=1)
