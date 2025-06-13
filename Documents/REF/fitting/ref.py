from calculating_dynamic_resolution import *
from convolution import convolution
from variables import *
from read_complex_matrix import *
import torch

def reflectometry(q, matrix):
    """
    Анализ спектра отражательной способности с поиском экстремумов
    
    Параметры:
    q: сетка волновых векторов
    matrix: матрица описания структуры
    
    Возвращает:
    #r: вектор всех значений коэффициента отражения
    #y: вектор всех значений коэффициента отражения со свёрткой
    """
    # Загрузка и расчет базовых матриц
    r, r_real, r_img = calculate_matrices_and_reflection(matrix, rough_res, kmax, q, Ndots, gap)

    # Свёртка
    r_conv = convolution(r, kmax, q, Ndots_norm, sigma1, sigma2, gap)
    
    return r, r_conv


def reflectometry_trace(q, matrix):
    """
    Анализ спектра отражательной способности с поиском экстремумов
    
    Параметры:
    q: сетка волновых векторов
    matrix: матрица описания структуры
    
    Возвращает:
    x_crit: критическая точка
    x_max/y_max: координаты максимумов
    x_min/y_min: координаты минимумов
    ----
    #r: вектор всех значений коэффициента отражения
    #y: вектор всех значений коэффициента отражения со свёрткой
    """
    # Загрузка и расчет базовых матриц
    r, r_real, r_img = calculate_matrices_and_reflection(matrix, rough_res, kmax, q, Ndots, gap)

    # Свёртка
    y = convolution(r, kmax, q, Ndots_norm, sigma1, sigma2, gap)
    
    # Поиск критической точки
    below_threshold = torch.where(r < 0.9)[0]
    if len(below_threshold) == 0:
        crit_dot = 0
    else:
        crit_dot = below_threshold[0]
    
    # Поиск локальных экстремумов
    left = r[:-2]
    center = r[1:-1]
    right = r[2:]
    
    max_mask = (center > left) & (center > right)
    min_mask = (center < left) & (center < right)
    
    # Фильтрация индексов
    raw_max_indices = torch.where(max_mask)[0] + 1
    raw_min_indices = torch.where(min_mask)[0] + 1
    
    max_indices = raw_max_indices[raw_max_indices > crit_dot]
    min_indices = raw_min_indices[raw_min_indices > crit_dot]
    
    # Получение координат и их 
    x_max = (q[max_indices])[0:Ndots_trace]
    y_max = (r[max_indices])[0:Ndots_trace]
    x_min = (q[min_indices])[0:Ndots_trace]
    y_min = (r[min_indices])[0:Ndots_trace]
    x_crit = q[crit_dot]

    
    return x_crit, y_max, x_max, y_min, x_min#, r, y