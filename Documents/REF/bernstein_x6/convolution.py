import torch
import math

def convolution(r, qmax, q0, Ndots_norm, sigma1_param, sigma2_param, gapi):
    """
    Выполняет свертку коэффициентов отражения с учетом параметров разрешения.
    
    Параметры:
    r : torch.Tensor
        Тензор коэффициентов отражения 
    qmax : float
        Максимальное значение волнового числа (определяет диапазон расчетов)
    q0 : torch.Tensor
        Тензор значений волнового числа 
    Ndots_norm : int
        Количество точек окна свертки
    sigma1_param, sigma2_param : torch.Tensor
        Параметры разрешения
    gapi : float
        Ширина переходной зоны между sigma1 и sigma2
        
    Возвращает:
    list[float]
        Список абсолютных значений коэффициентов отражения

    Примечания:
    1. Все вычисления оптимизированы для работы с GPU
    2. Используется адаптивная нормировка весовых коэффициентов
    3. Автоматическая обработка разных устройств (CPU/GPU)
    4. Гарантируется ограничение индексов в допустимых пределах
    """
    # Определение устройства выполнения (GPU/CPU)
    device ='cpu'
    
    qmax = qmax if isinstance(qmax, torch.Tensor) else torch.tensor(qmax, device=device, dtype=torch.float32)
    
    # Конвертация q0 в тензор с проверкой текущего устройства
    if not isinstance(q0, torch.Tensor):
        q0 = torch.tensor(q0, device=device, dtype=torch.float32)
    elif q0.device.type != device:
        q0 = q0.to(device=device)
    
    sigma1_scaled = sigma1_param * 0.4191  
    sigma2_scaled = sigma2_param * 0.4191
    gap = torch.tensor(gapi, device=device, dtype=torch.float32)  # Тензор зазора
    
    # Вычисление относительных значений q
    q_rel = q0 / qmax  # Нормировка на максимальное значение
    
    # Создание базового окна для свертки
    q_window = torch.linspace(-3, 3, Ndots_norm, device=device).unsqueeze(0)
    
    # Инициализация весовых коэффициентов нормальным распределением
    weights = torch.exp(-0.5 * q_window.pow(2)) * 0.39894228
    weights = weights.repeat(q0.size(0), 1)  # Расширение для всех точек
    

    # Расчет матрицы сигм с использованием функции разрешения
    sigma_matrix = resolution_function_smooth_step(
        torch.ones_like(q_rel, device=device), 
        q_rel,  
        sigma1_scaled,  
        sigma2_scaled,  
        gap     
    ).unsqueeze(1)  # Добавление размерности для broadcast
      
    # Вычисление абсолютных значений q с учетом окна
    q_abs = q0.unsqueeze(1) * (1 + sigma_matrix * q_window)
        # Поиск позиций в исходном массиве через бинарный поиск
    indices = torch.searchsorted(q0, q_abs)
    
        # Ограничение индексов в допустимом диапазоне
    indices = torch.clamp(indices, 0, len(r)-1)
        # Создание маски валидных значений
    valid_mask = (q_abs >= q0[0]) & (q_abs <= q0[-1])
        # Применение маски к весам
    weights = weights * valid_mask  
        
        # Нормировка весов с защитой от деления на ноль
    sum_weights = weights.sum(dim=1, keepdim=True)
    weights = torch.where(sum_weights > 0, weights/sum_weights, weights)
    
    # Возврат результата свертки в виде списка
    return torch.sum(r[indices] * weights, dim=1)


def resolution_function_smooth_step(qmax_rel, q_rel, sigma1, sigma2, gap):
    """
    Векторизованная реализация функции разрешения с плавным переходом между сигмами.
    
    Параметры:
    qmax_rel : torch.Tensor
        Максимальное относительное значение q (не используется в текущей реализации)
    q_rel : torch.Tensor
        Относительные значения q (нормированные на qmax)
    sigma1, sigma2 : torch.Tensor
        Параметры разрешения
    gapi : float
        Ширина переходной зоны между sigma1 и sigma2
        
    Возвращает:
    torch.Tensor
        Тензор значений сигм для каждого q_rel
    """
    # Все вычисления с использованием тензорных операций
    transition_start = 0.5 - gap/2  
    transition_end = 0.5 + gap/2    
    
    # Создаем маски
    mask_after = q_rel > transition_end
    mask_transition = (q_rel > transition_start) & (q_rel <= transition_end)
    mask_before = ~(mask_after | mask_transition)
    
    t = (q_rel - transition_start) / gap
    t = torch.clamp(t, 0.0, 1.0)
    
    # Собираем результат
    result = torch.zeros_like(q_rel)
    result = torch.where(mask_before, sigma1, result)
    result = torch.where(mask_transition, sigma1*(1-t) + sigma2*t, result)
    result = torch.where(mask_after, sigma2, result)
    
    return result