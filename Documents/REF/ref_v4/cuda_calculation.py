import cupy as cp
import cmath as m
from transform_array import *

def calculate_matrices_and_reflection(matrix, rough_res, Ndots):
    """
    Расчет матриц и коэффициента отражения для оптической системы на GPU.
    
    Параметры аналогичны CPU версии.
    """
    # Перенос данных на GPU и инициализация
    transformed_array = transform_array(matrix, rough_res, lambda z: z)
    ro = cp.array(transformed_array)[:, 1]
    d = cp.array(transformed_array)[:, 0]
    
    # Создание пустых матриц на GPU
    q = cp.empty((len(ro), Ndots), dtype=complex)
    dm = cp.empty((len(ro), Ndots, 2, 2), dtype=complex)
    dmi = cp.empty((len(ro), Ndots, 2, 2), dtype=complex)
    pm = cp.empty((len(ro), Ndots, 2, 2), dtype=complex)
    M = cp.empty((Ndots, 2, 2), dtype=complex)
    
    # Расчет базовых параметров
    dk0 = m.sqrt(m.sqrt(float(cp.mean(ro**2)))/Ndots)
    Pe = cp.array([[1, 0], [0, 1]])
    
    # Создание сетки на GPU
    i_indices = cp.arange(Ndots, dtype=float) + 0.001
    j_indices = cp.arange(len(ro))
    j_grid, i_grid = cp.meshgrid(j_indices, i_indices, indexing='ij')
    
    # Расчет q и матриц
    q = cp.sqrt((i_grid * dk0)**2 - 12.56637 * ro[j_grid])
    
    # Заполнение матрицы dm
    ones_matrix = cp.ones((len(ro), Ndots))
    dm[j_grid, i_grid.astype(int), :, :] = cp.array([
        [ones_matrix, ones_matrix],
        [q, -q]
    ]).transpose(2, 3, 0, 1)
    
    # Расчет фазовых компонентов
    phase = q * d[j_grid]
    exp_neg = cp.exp(1j * (-1 * phase))
    exp_pos = cp.exp(1j * phase)
    zeros = cp.zeros_like(q)
    
    # Заполнение фазовой матрицы
    pm[j_grid, i_grid.astype(int), :, :] = cp.array([
        [exp_neg, zeros],
        [zeros, exp_pos]
    ]).transpose(2, 3, 0, 1)
    
    # Расчет обратных матриц
    dmi = cp.linalg.inv(dm.reshape(-1, 2, 2)).reshape(len(ro), Ndots, 2, 2)
    
    # Расчет итоговой матрицы M
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
    
    # Возвращаем результаты, преобразуя их обратно в numpy массивы
    return cp.asnumpy(reflection_coefficient), dk0

def calculate_matrices_and_reflection_with_gpu(matrix, rough_res, Ndots):
    """
    Обертка для обработки ошибок и управления памятью GPU
    """
    try:
        with cp.cuda.Device(0):  # Использовать первое доступное GPU устройство
            return calculate_matrices_and_reflection_cuda(matrix, rough_res, Ndots)
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Error: {e}")
        print("Falling back to CPU calculation...")
        return calculate_matrices_and_reflection(matrix, rough_res, Ndots)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise