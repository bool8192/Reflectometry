import numpy as np
from scipy.interpolate import interp1d
from pykeops.numpy import LazyTensor


def keops_interpolate(q, r, n_points, bandwidth):
    """
    Интерполяция через PyKeOps
    
    Параметры
    ---------
    q : np.ndarray
        Нерегулярная отсортированная сетка (1D массив).
    r : np.ndarray
        Значения в точках q (1D массив).
    n_points : int
        Количество точек для интерполяции.
    bandwidth : float
        Параметр ядра (ширина радиальной функции).
    """
    # Создаем регулярную целевую сетку
    q_new = np.linspace(q.min(), q.max(), n_points)[:, None]  # (n_points, 1)
    
    # Преобразуем данные в LazyTensor
    q_i = LazyTensor(q_new.astype('float32'))  # Целевые точки (n_points, 1)
    q_j = LazyTensor(q.astype('float32')[None, :])  # Исходные точки (1, n_data)
    r_j = LazyTensor(r.astype('float32')[None, :])  # Значения (1, n_data)
    
    # Вычисляем расстояния между точками
    D_ij = ((q_i - q_j) ** 2).sqrt()  # (n_points, n_data)
    
    # Применяем ядро (радиальная базисная функция)
    K_ij = (-D_ij / bandwidth).exp()  # (n_points, n_data)
    
    # Интерполяция: взвешенная сумма значений
    r_new = (K_ij * r_j).sum(dim=1) / K_ij.sum(dim=1)  # (n_points,)
    
    # Конвертируем результат в NumPy
    r_new = r_new.numpy().flatten()
    
    return q_new, r_new


def scipy_interpolate(q, r, n_points, kind):
    """
    Интерполяция через SciPy
    
    Параметры
    ---------
    q : array-like
        Нерегулярная отсортированная сетка (1D массив).
    r : array-like
        Значения в точках q (1D массив).
    n_points : int
        Количество точек для интерполяции.
    kind : str
        Метод интерполяции: 'linear', 'nearest', 'cubic', 'quadratic'
    
    Возвращает
    ----------
    q_new : np.ndarray
        Новая регулярная сетка.
    r_new : np.ndarray
        Интерполированные значения.
    """
    # Преобразуем входные данные в массивы NumPy с явным указанием типа float64
    q = np.asarray(q, dtype=np.float64).squeeze()  # Удаляем лишние измерения
    r = np.asarray(r, dtype=np.float64).squeeze()
    
    # Проверка на наличие NaN или Inf
    if np.isnan(q).any() or np.isnan(r).any():
        raise ValueError("Массивы q и r не должны содержать NaN.")
    if np.isinf(q).any() or np.isinf(r).any():
        raise ValueError("Массивы q и r не должны содержать бесконечности.")
    
    # Проверка размерностей
    if q.ndim != 1 or r.ndim != 1:
        raise ValueError("q и r должны быть одномерными массивами.")
    
    # Создаем интерполяционную функцию
    interp_func = interp1d(
        q, 
        r, 
        kind=kind, 
        fill_value='extrapolate'
    )
    
    # Генерируем регулярную сетку
    q_new = np.linspace(q.min(), q.max(), n_points)
    r_new = interp_func(q_new)
    
    return q_new, r_new