from scipy.stats import norm
import numpy as np

def convolution(r, qmax, q0, Ndots_norm, sigma1, sigma2, gapi):
    Ndots = len(r)
    
    # Массив относительных q (нормировка к максимальному значению)
    q_rel = q0 / qmax
    
    # Вычисляем сигмы для всех точек (уже в относительных единицах)
    sigma_matrix = np.array([resolution_function_smooth_step(1.0, q, sigma1, sigma2, gapi) 
                           for q in q_rel])[:, np.newaxis]
    
    # Динамическое окно для каждого q (относительные отклонения)
    q_window = np.linspace(-3, 3, Ndots_norm)[np.newaxis, :]  # В сигмах
    
    # Абсолютные q-значения для свертки
    q_abs = q0[:, np.newaxis] * (1 + sigma_matrix * q_window)
    
    # Поиск ближайших индексов с защитой границ
    indices = np.clip(np.searchsorted(q0, q_abs), 0, Ndots-1)
    
    # Вычисление весов
    weights = norm.pdf(q_window, 0, 1)  
    weights = np.repeat(weights, len(q0), axis=0)  # Повторяем для всех точек
    
    # Маска валидных точек (в пределах исходного диапазона)
    valid_mask = (q_abs >= q0[0]) & (q_abs <= q0[-1])
    weights *= valid_mask
    
    # Нормировка весов для каждого q
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    sum_weights[sum_weights == 0] = 1.0  # Защита от деления на ноль
    weights /= sum_weights
    
    # Применяем свертку
    return np.sum(r[indices] * weights, axis=1)

def resolution_function_smooth_step(qmax_rel, q_rel, sigma1, sigma2, gap):
    transition_start = 0.5 - gap/2
    transition_end = 0.5 + gap/2
    
    if q_rel > transition_end:
        return sigma2
    elif q_rel > transition_start:
        t = (q_rel - transition_start) / (gap)
        return sigma1 * (1 - t) + sigma2 * t
    else:
        return sigma1
