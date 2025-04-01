from scipy.stats import norm
import numpy as np

def convolution(r, sigma, q0, Ndots_norm):
    """
    Применяет гауссову свертку к массиву комплексных чисел
    
    Parameters:
    r (np.array): входной массив комплексных коэффициентов отражения
    sigma (float): относительная ширина гауссиана
    Ndots_norm (int): количество точек для построения нормального распределения
    
    Returns:
    np.array: преобразованный массив комплексных чисел
    """
    Ndots = len(r)
    
    # Создаём веса для окрестных точек
    q_window = np.linspace(1-3*sigma, 1+3*sigma, Ndots_norm)
    
    # Создаём матрицу весов
    weights = norm.pdf(q_window, 1, sigma)
    weights = weights/np.sum(weights)
    
    # Создаём матрицу индексов для всех точек
    q_matrix = q0[:, np.newaxis] * q_window
    indices_matrix = np.searchsorted(q0, q_matrix)
    
    # Маска для индексов, выходящих за границы
    mask = indices_matrix < Ndots
    indices_matrix = indices_matrix * mask
    
    # Создаём маску весов соответствующего размера
    weights_matrix = np.tile(weights, (Ndots, 1)) * mask
    
    # Получаем значения r для всех индексов
    r_matrix = r[indices_matrix] * mask
    
    # Считаем взвешенное среднее
    return np.sum(r_matrix * weights_matrix, axis=1)