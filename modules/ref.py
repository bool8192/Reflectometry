from calculating_dynamic_resolution import *
from convolution import convolution
from variables import *
from read_complex_matrix import *
import torch

def reflectometry(q, matrix, var_sigma1, var_sigma2, var_I0, var_Ibkg):
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
    #print("I0 inside reflectometry, grad_fn:", var_I0.grad_fn)
    #print("Ibkg inside reflectometry, grad_fn:", var_Ibkg.grad_fn)
    #print("sigmas inside reflectometry, grad_fn:", var_sigma1.grad_fn, "  ", var_sigma2.grad_fn)
    
    r_conv = convolution(r, kmax, q, Ndots_norm, var_sigma1, var_sigma2, gap) + var_Ibkg/var_I0
    
    #print("r_conv sample grad_fn:", r_conv[0].grad_fn)
    
    return r, r_conv