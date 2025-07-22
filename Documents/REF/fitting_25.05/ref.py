import numpy as np
from calculating_dynamic_resolution import *
from convolution import convolution
from variables import *
from read_complex_matrix import *


def reflectometry_trace(rough1, len1, rough2, len2):
    """
    В коде функции обращаемся к параметрам как глобальным переменным
    Анализ спектра отражательной способности с поиском экстремумов
    
    Параметры:
    kmax: максимальное значение волнового вектора
    gap: относительная длина перехода между разрешениями
    Ndots: количество точек сетки по дефолту
    Ndots_norm: количество точек в нормальном распределение свёртки
    sigma1, sigma2: разрешение установки 
    rough_res: число переходных слоёв шероховатости
    
    Возвращает:
    x_crit: критическая точка
    x_max/y_max: координаты максимумов
    x_min/y_min: координаты минимумов
    ----
    q: вектор всех значений волнового вектора
    r: вектор всех значений коэффициента отражения
    y: вектор всех значений коэффициента отражения со свёрткой
    """
    # Загрузка и расчет базовых матриц
    matr = read_complex_matrix('ref_matrix.txt')
    matr[1,0] = len1
    matr[1,2] = rough1
    matr[2,0] = len2
    matr[2,2] = rough2
    q = dynamic_mesh_q(kmax, Ndots, gap)
    r, r_real, r_img = calculate_matrices_and_reflection(matr, rough_res, kmax, q, Ndots, gap)
    
    # Свёртка
    y = convolution(np.array(r), kmax, q, Ndots_norm, sigma1, sigma2, gap)
    
    # Поиск критической точки
    below_threshold = np.where(y < 0.9)[0]
    if len(below_threshold) == 0:
        crit_dot = 0
    else:
        crit_dot = below_threshold[0]
    
    # Поиск локальных экстремумов
    left = y[:-2]
    center = y[1:-1]
    right = y[2:]
    
    max_mask = (center > left) & (center > right)
    min_mask = (center < left) & (center < right)
    
    # Фильтрация индексов
    raw_max_indices = np.where(max_mask)[0] + 1
    raw_min_indices = np.where(min_mask)[0] + 1
    
    max_indices = raw_max_indices[raw_max_indices > crit_dot]
    min_indices = raw_min_indices[raw_min_indices > crit_dot]
    
    # Получение координат и их 
    x_max = (q[max_indices])[0:Ndots_trace]
    y_max = (y[max_indices])[0:Ndots_trace]
    x_min = (q[min_indices])[0:Ndots_trace]
    y_min = (y[min_indices])[0:Ndots_trace]
    x_crit =q[crit_dot]

    
    return x_crit, x_max, y_max, x_min, y_min #, q, r, y
