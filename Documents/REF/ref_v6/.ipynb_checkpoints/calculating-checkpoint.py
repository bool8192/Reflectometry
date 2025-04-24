import numpy as np
import cmath as m
from transform_array import *
from variables import *

def calculate_matrices_and_reflection(matrix):
    """
    Расчет матриц и коэффициента отражения для оптической системы.

    Параметры
    ---------
    matrix : numpy.ndarray
        Входная матрица с параметрами слоев
        
    Возвращает
    ----------
    numpy.ndarray
        Массив коэффициентов отражения
    """
    # Инициализация базовых массивов
    ro = np.array(transform_array(matrix, rough_res, lambda z: z))[:, 1]
    d = np.array(transform_array(matrix, rough_res, lambda z: z))[:,0]

    # Расчет базовых параметров
    dk0 = dq/Ndots_norm
    Ndots = int(kmax/dk0)
    Pe = np.array([[1, 0], [0, 1]])
    
    # Создание пустых матриц
    q = np.empty((len(ro), Ndots), dtype=complex)
    dm = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    dmi = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    pm = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    M = np.empty((Ndots, 2,2), dtype=complex)

   # Создание сетки
    i_indices = np.arange(Ndots) + 0.999
    j_indices = np.arange(len(ro))
    j_grid, i_grid = np.meshgrid(j_indices, i_indices, indexing='ij')
    
    # Расчет q и матриц
    q = np.sqrt((i_grid * dk0)**2 - 12.56637 * ro[j_grid])
    dm[j_grid, i_grid.astype(int), :, :] = np.array([
        [np.ones((len(ro), Ndots)), np.ones((len(ro), Ndots))],
        [q, -q]
    ]).transpose(2, 3, 0, 1)
    
    # Расчет фазовых компонентов
    phase = q * d[j_grid]
    exp_neg = np.exp(1j * (-1 * phase))
    exp_pos = np.exp(1j * phase)
    zeros = np.zeros_like(q)
    
    # Заполнение фазовой матрицы
    pm[j_grid, i_grid.astype(int), :, :] = np.array([
        [exp_neg, zeros],
        [zeros, exp_pos]
    ]).transpose(2, 3, 0, 1)

    # Расчет обратных матриц
    dmi = np.linalg.inv(dm.reshape(-1, 2, 2)).reshape(len(ro), Ndots, 2, 2)
    
    # Расчет итоговой матрицы M в зависимости от количества слоев
    if dm.shape[0] == 2:
        M = dmi[0] @ dm[1]
    elif dm.shape[0] == 3:
        M = dmi[0] @ dm[1] @ pm[1] @ dmi[1] @ dm[2]
    else:
        for j in range(1, dm.shape[0]-1):
            Pe = dm[j] @ pm[j] @ dmi[j] @ Pe
        M = dmi[0] @ Pe @ dm[-1]
    
    # Расчет коэффициента отражения
    r = (M[:,1,0]/M[:,0,0])
    r_real = []
    r_img = []
    r_abs = []
    
    for i in range(0, Ndots-1):
        r_real.append(r[i].real)
        r_img.append(r[i].imag)
        r_abs.append((m.polar(r[i])[0])**2) 
        
    return r_abs, r_real, r_img, i_indices*dk0