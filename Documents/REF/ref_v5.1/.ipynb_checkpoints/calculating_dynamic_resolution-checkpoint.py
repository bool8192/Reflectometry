import numpy as np
from variables import *
from transform_array import *


def resolution_function(qmax, q, sigma1, sigma2):
  sigma = sigma1 + (sigma2-sigma1)*(q/qmax)
  return sigma


def resolution_function_step(qmax, q, sigma1, sigma2):
  if q > qmax/2: 
      sigma = sigma2
  else:
      sigma = sigma1
  return sigma


def resolution_function_smooth_step(qmax, q, sigma1, sigma2):
  if q > 0.55*qmax: 
      sigma = sigma2
  elif (q <= 0.55*qmax and q > 0.45*qmax):
      sigma = sigma1 + (sigma2-sigma1)*(q/qmax)
  else:
      sigma = sigma1
  return sigma

def dynamic_mesh_q(kmax, Ndots):
    """
    Генерация разбиения = секти с динамическим шагом;
    принципиально не векторизуемо, так как вычисляется рекурсивно.

    Параметры
    ---------
    kmax : float
        Значение волнового вектора, вплоть до которого генерируем разбиение 

    Возвращает
    ----------
    numpy.ndarray
        Массив q0
    """
    q0 = [kmax/Ndots]
    while q0[-1] < kmax:
        q0.append(q0[-1]+5.9*q0[-1]*resolution_function(kmax, q0[-1], sigma1, sigma2)/Ndots_norm)
    return np.array(q0)


def calculate_matrices_and_reflection(matrix, rough_res, kmax, q, Ndots):
    """
    Расчет матриц и коэффициента отражения для оптической системы.

    Параметры
    ---------
    matrix : numpy.ndarray
        Входная матрица с параметрами слоев
    rough_res : float
        Параметр разрешения
    kmax : float
        Максимальное значение k

    Возвращает
    ----------
    numpy.ndarray
        Массив коэффициентов отражения
    """
    # Инициализация базовых массивов
    ro = np.array(transform_array(matrix, rough_res, lambda z: z))[:, 1]
    d = np.array(transform_array(matrix, rough_res, lambda z: z))[:,0]

    # Расчет базовых параметров
    q0 = dynamic_mesh_q(kmax, Ndots)
    Ndots = len(q0)
    Pe = np.array([[1, 0], [0, 1]])

    # Создание пустых матриц
    dm = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    dmi = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    pm = np.empty((len(ro), Ndots, 2,2), dtype=complex)
    M = np.empty((Ndots, 2,2), dtype=complex)

    # Создание сетки
    i_indices = np.arange(Ndots) + 0.001
    j_indices = np.arange(len(ro))
    j_grid, i_grid = np.meshgrid(j_indices, i_indices, indexing='ij')
    j_grid2, i_grid2 = np.meshgrid(j_indices, q0, indexing='ij')

    # Расчет q и матриц
    q = np.sqrt((i_grid2)**2 - 12.56637 * ro[j_grid2])
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
    reflection_coefficient = M[:,1,0]/M[:,0,0]
    return reflection_coefficient
