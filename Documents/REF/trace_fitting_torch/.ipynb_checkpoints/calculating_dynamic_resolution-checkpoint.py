import torch
import cmath as m
from variables import *
from add_roughness import transform_array


def resolution_function_smooth_step(qmax, q, sigma1, sigma2, gap):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  qmax = qmax/2
  if gap != 0:
      if q > (0.5+gap/2)*qmax: 
          sigma = sigma2
      elif (q <= (0.5+gap/2)*qmax and q > (0.5-gap/2)*qmax):
          sigma = (q-qmax/2)*(sigma2-sigma1)/(qmax*gap) + (sigma2+sigma1)/2
      else:
          sigma = sigma1
  else:
      if q > qmax/2:
          sigma = sigma2
      else:
          sigma = sigma1
  return sigma


def dynamic_mesh_q(kmax, Ndots, gap):
    """
    Генерация разбиения = секти с динамическим шагом;
    принципиально не векторизуемо, так как вычисляется рекурсивно.

    Параметры
    ---------
    kmax : float
        Значение волнового вектора, вплоть до которого генерируем разбиение 

    Возвращает
    ----------
    torch.Tensor
        Массив q0
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q0 = [kmax/Ndots]
    while q0[-1] < kmax*0.5:
        if (5.9*q0[-1]*resolution_function_smooth_step(kmax, q0[-1], sigma1, sigma2, gap)/Ndots_norm) > kmax/Ndots:
            q0.append(q0[-1]+5.9*q0[-1]*resolution_function_smooth_step(kmax, q0[-1], sigma1, sigma2, gap)/Ndots_norm)
        else:
            q0.append(q0[-1]+kmax/Ndots)
    return 2*torch.tensor(q0, dtype=torch.double, device=device).requires_grad_(True)


def calculate_matrices_and_reflection(matrix, rough_res, kmax, q, Ndots, gap):
    """
    Вычисление матриц переноса и коэффициента отражения для многослойной оптической системы.
    Все операции выполняются с использованием тензоров PyTorch.

    Параметры
    ---------
    matrix : torch.Tensor
        Тензор размерности [N×3] с параметрами слоев:
        - d (толщина), ro (плотность длины рассеяния), rough (шероховатость)
    rough_res : int
        Разрешение по шероховатости (влияет на преобразование параметров)
    kmax : int
        Максимальное значение волнового вектора
    q : float
        сетка волновых векторов
    Ndots : int
        Количество точек дискретизации для q0
    gap : float
        зазор перехода между режимами с разным разрешением

    Возвращает
    ----------
    r_abs : list[float]
        Квадраты модулей коэффициентов отражения (интенсивность)
    r_real : list[float]
        Вещественные части комплексных коэффициентов отражения
    r_img : list[float]
        Мнимые части комплексных коэффициентов отражения

    Примечания
    ----------
    Алгоритм использует матричный метод для расчета многослойных систем:
    1. Преобразование входных параметров с учетом шероховатостей
    2. Генерация адаптивной сетки волновых чисел
    3. Послойный расчет матриц интерфейсов и распространения
    4. Композиция полной матрицы системы
    5. Вычисление коэффициентов отражения из результирующей матрицы
    """    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Преобразование входных данных в тензоры PyTorch
    if isinstance(matrix, torch.Tensor):
        matrix_tensor = matrix.detach().clone()
    else:
        matrix_tensor = torch.tensor(matrix, dtype=torch.float64)
    
    # Преобразование массива с использованием transform_array (предполагается, что он работает с тензорами)
    transformed = transform_array(matrix_tensor, rough_res)
    if isinstance(transformed, torch.Tensor):
        transformed = transformed.detach().clone().type(torch.complex128)
    else:
        transformed = torch.tensor(transformed, dtype=torch.complex128)
    
    ro = transformed[:, 1]
    d = transformed[:, 0]

    # Генерация q0 с использованием torch
    q0 = q * 0.50
    Ndots = q0.numel()
    
    # Инициализация матриц
    Pe = torch.eye(2, dtype=torch.complex128, device=device)
    
    # Создание тензоров с явным указанием устройства
    dm = torch.zeros((len(ro), Ndots, 2, 2), dtype=torch.complex128, device=device)
    pm = torch.zeros_like(dm)
    M = torch.zeros((Ndots, 2, 2), dtype=torch.complex128, device=device)
    
    # Создание сеток индексов
    j_indices = torch.arange(len(ro), device=device)
    i_indices = torch.arange(Ndots, device=device)
    
    # Векторизованные вычисления
    j_grid, i_grid = torch.meshgrid(j_indices, i_indices, indexing='ij')
    q0_expanded = q0[i_grid]
    
    # Вычисление q_values с обработкой комплексных чисел
    ro_expanded = ro[j_grid.long()]
    under_sqrt = (q0_expanded.to(torch.complex128))**2 - 12.56637 * ro_expanded
    q_values = torch.sqrt(under_sqrt)
    
    # Заполнение матрицы dm
    dm[..., 0, 0] = 1.0
    dm[..., 0, 1] = 1.0
    dm[..., 1, 0] = q_values
    dm[..., 1, 1] = -q_values
    
    # Фазовые вычисления
    d_expanded = d[j_grid.long()]
    phase = q_values * d_expanded
    exp_neg = torch.exp(-1j * phase)
    exp_pos = torch.exp(1j * phase)
    
    # Заполнение фазовой матрицы pm
    pm[..., 0, 0] = exp_neg
    pm[..., 1, 1] = exp_pos
    
    # Вычисление обратных матриц
    dmi = torch.linalg.inv(dm)
    
    # Векторизованное умножение матриц
    if len(ro) == 2:
        M = torch.einsum('nij,njk->nik', dmi[0], dm[1])
    elif len(ro) == 3:
        M = torch.einsum('nij,njk,nkl,nlm->nim', dmi[0], dm[1], pm[1], dmi[1], dm[2])
    else:
        Pe = torch.eye(2, dtype=torch.complex128, device=device).repeat(Ndots, 1, 1)
        for j in range(1, len(ro)-1):
            Pe = torch.einsum('nij,njk,nkl->nil', dm[j], pm[j], dmi[j]) @ Pe
        M = dmi[0] @ Pe @ dm[-1]
    
    # Вычисление коэффициента отражения
    r = M[:, 1, 0] / M[:, 0, 0]
    real = torch.nan_to_num(r.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = torch.nan_to_num(r.imag, nan=0.0, posinf=0.0, neginf=0.0)
    r = torch.complex(real, imag)
    
    # Расчет результатов
    r_abs = torch.abs(r).pow(2).tolist()
    r_real = r.real.tolist()
    r_img = r.imag.tolist()
    
    return r_abs, r_real, r_img