from calculating_dynamic_resolution import *
from convolution import convolution
from variables import *
from read_complex_matrix import *
import torch
import numpy as np

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
    rt = torch.tensor(r)

    # Свёртка
    r_conv = torch.tensor(convolution(r, kmax, q, Ndots_norm, sigma1, sigma2, gap))
    
    return rt, r_conv