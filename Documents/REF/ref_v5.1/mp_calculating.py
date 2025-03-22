import numpy as np
import cmath as m
from transform_array import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_matrices_and_reflection(matrix, rough_res, Ndots):
    """
    Расчет матриц и коэффициента отражения для оптической системы.
    Параметры
    ---------
    matrix : numpy.ndarray
        Входная матрица с параметрами слоев
    rough_res : float
        Параметр разрешения
    Ndots : int
        Количество точек дискретизации
    Возвращает
    ----------
    numpy.ndarray
        Массив коэффициентов отражения
    """
    # Инициализация базовых массивов
    ro = np.array(transform_array(matrix, rough_res, lambda z: z))[:, 1]
    d = np.array(transform_array(matrix, rough_res, lambda z: z))[:, 0]
  
    # Создание пустых матриц
    q = np.empty((len(ro), Ndots), dtype=complex)
    dm = np.empty((len(ro), Ndots, 2, 2), dtype=complex)
    dmi = np.empty((len(ro), Ndots, 2, 2), dtype=complex)
    pm = np.empty((len(ro), Ndots, 2, 2), dtype=complex)
    M = np.empty((Ndots, 2, 2), dtype=complex)

    # Расчет базовых параметров
    dk0 = m.sqrt(m.sqrt(np.mean(ro**2)) / Ndots)
    Pe = np.array([[1, 0], [0, 1]])

    # Создание сетки
    i_indices = np.arange(Ndots) + 0.001
    j_indices = np.arange(len(ro))
    j_grid, i_grid = np.meshgrid(j_indices, i_indices, indexing='ij')

    # Расчет q и матриц
    q = np.sqrt((i_grid * dk0) ** 2 - 12.56637 * ro[j_grid])

    def compute_dm_pm(j):
        dm[j, i_grid.astype(int), :, :] = np.array([
            [np.ones(Ndots), np.ones(Ndots)],
            [q[j], -q[j]]
        ]).transpose(2, 0, 1)

        phase = q[j] * d[j]
        exp_neg = np.exp(1j * (-1 * phase))
        exp_pos = np.exp(1j * phase)
        zeros = np.zeros(Ndots, dtype=complex)

        pm[j, i_grid.astype(int), :, :] = np.array([
            [exp_neg, zeros],
            [zeros, exp_pos]
        ]).transpose(2, 0, 1)

        return dm[j], pm[j]

    # Параллельное вычисление
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(compute_dm_pm, j) for j in range(len(ro))]
        for future in as_completed(futures):
            _dm, _pm = future.result()

    # Расчет обратных матриц
    for j in range(len(ro)):
        dmi[j] = np.linalg.inv(dm[j])

    # Расчет итоговой матрицы M в зависимости от количества слоев
    if dm.shape[0] == 2:
        M = dmi[0] @ dm[1]
    elif dm.shape[0] == 3:
        M = dmi[0] @ dm[1] @ pm[1] @ dmi[1] @ dm[2]
    else:
        for j in range(1, dm.shape[0] - 1):
            Pe = dm[j] @ pm[j] @ dmi[j] @ Pe
        M = dmi[0] @ Pe @ dm[-1]

    # Расчет коэффициента отражения
    reflection_coefficient = M[:, 1, 0] / M[:, 0, 0]
    return reflection_coefficient, dk0