import torch
import cmath as m
from variables import *
from add_roughness import transform_array
from functools import reduce
from functools import partial


def resolution_function_smooth_step(qmax, q, sigma1, sigma2, gap):
  device = 'cpu'
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
    device = 'cpu'
    q0 = [kmax/Ndots]
    while q0[-1] < kmax*0.5:
        if (5.9*q0[-1]*resolution_function_smooth_step(kmax, q0[-1], sigma1, sigma2, gap)/Ndots_norm) > kmax/Ndots:
            q0.append(q0[-1]+5.9*q0[-1]*resolution_function_smooth_step(kmax, q0[-1], sigma1, sigma2, gap)/Ndots_norm)
        else:
            q0.append(q0[-1]+kmax/Ndots)
    return 2*torch.tensor(q0, dtype=torch.double, device=device)


def calculate_matrices_and_reflection(matr, rough_res, kmax, q, Ndots, gap):
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

    device = 'cpu'

    scaling_factors = torch.tensor(
    [1e-10, 1e+14, 1e-10, 1.0, 1.0, 1.0],
    device=matr.device,
    dtype=matr.dtype
    )
    matrix_tensor = matr * scaling_factors

    transformed = transform_array(matrix_tensor, rough_res).to(device)

    ro = transformed[:, 1] * 1.0e-14
    z = transformed[:, 0] * 1.0e+10
    layers = torch.cat((z.reshape(-1, 1), ro.real.reshape(-1, 1), ro.imag.reshape(-1, 1)), dim=1)

    xx = torch.as_tensor(q * 0.5e-10, dtype=torch.complex128, device=device).view(-1)
    nlayers = layers.size(0) - 2
    npnts = xx.size(0)
    kn = torch.zeros((npnts, nlayers + 2), dtype=torch.complex128, device=device)
    sld = torch.zeros(nlayers + 2, dtype=torch.complex128, device=device)
    sld[1:] += ((layers[1:, 1] - layers[0, 1]) + 1j * (torch.abs(layers[1:, 2]) + 1e-12)) * 1.0e-6
    kn[:] = torch.sqrt((xx[:, None] ** 2.0) / 4.0 - 4.0 * 3.14159 * sld)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:])
    if nlayers > 0:
        M = torch.zeros((npnts, nlayers + 1, 2, 2), dtype=torch.complex128, device=device)
        mi00 = torch.ones((npnts, nlayers + 1), dtype=torch.complex128, device=device)
        mi00[:, 1:] = torch.exp(kn[:, 1:-1] * 1j * torch.abs(layers[1:-1, 0]))
        mi11 = 1.0 / mi00
        mi10 = rj * mi00
        mi01 = rj * mi11
        M[:, :, 0, 0] = mi00
        M[:, :, 0, 1] = mi01
        M[:, :, 1, 0] = mi10
        M[:, :, 1, 1] = mi11

        # Список батч-матриц
        M_list = list(torch.unbind(M, dim=1))  # каждый элемент shape (npnts,2,2)

        # Если нечётное число, добавим единичную матрицу (по npnts)
        I = torch.eye(2, dtype=M.dtype, device=device).unsqueeze(0).expand(npnts, -1, -1)
        while len(M_list) > 1:
            if len(M_list) % 2 == 1:
                M_list.append(I)
            M_list = [torch.matmul(M_list[i + 1], M_list[i]) for i in range(0, len(M_list), 2)]

        Mtot = M_list[0]  # shape (npnts,2,2)
        mrtot00 = Mtot[:, 0, 0]
        mrtot01 = Mtot[:, 0, 1]
        mrtot10 = Mtot[:, 1, 0]
        mrtot11 = Mtot[:, 1, 1]
    else:
        mrtot00 = torch.ones(npnts, dtype=torch.complex128, device=device)
        mrtot01 = rj[:, 0]

    r = mrtot01 / mrtot00
    reflectivity = r * torch.conj(r)

    # Расчет результатов
    r_real = r.real
    r_img = r.imag

    return reflectivity, r_real, r_img


def calculate_matrices_and_reflection_freeform(matr, rough_res, kmax, q, Ndots, gap):

    device = 'cpu'

    scaling_factors = torch.tensor(
        [1e-10, 1e+14, 1e-10],
        device=matr.device,
        dtype=matr.dtype
    )
    matrix_tensor = matr * scaling_factors

    transformed = transform_array(matrix_tensor, rough_res).to(device)

    ro = transformed[:, 1] * 1.0e-14
    z = transformed[:, 0] * 1.0e+10
    layers = torch.cat((z.reshape(-1, 1), ro.real.reshape(-1, 1), ro.imag.reshape(-1, 1)), dim=1)

    xx = torch.as_tensor(q * 1e-10, dtype=torch.complex128, device=device).view(-1)
    nlayers = layers.size(0) - 2
    npnts = xx.size(0)
    kn = torch.zeros((npnts, nlayers + 2), dtype=torch.complex128, device=device)
    sld = torch.zeros(nlayers + 2, dtype=torch.complex128, device=device)
    sld[1:] += ((layers[1:, 1] - layers[0, 1]) + 1j * (torch.abs(layers[1:, 2]) + 1e-12)) * 1.0e-6
    kn[:] = torch.sqrt((xx[:, None] ** 2.0) / 4.0 - 4.0 * 3.14159 * sld)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:])
    if nlayers > 0:
        M = torch.zeros((npnts, nlayers + 1, 2, 2), dtype=torch.complex128, device=device)
        mi00 = torch.ones((npnts, nlayers + 1), dtype=torch.complex128, device=device)
        mi00[:, 1:] = torch.exp(kn[:, 1:-1] * 1j * torch.abs(layers[1:-1, 0]))
        mi11 = 1.0 / mi00
        mi10 = rj * mi00
        mi01 = rj * mi11
        M[:, :, 0, 0] = mi00
        M[:, :, 0, 1] = mi01
        M[:, :, 1, 0] = mi10
        M[:, :, 1, 1] = mi11

        # Список батч-матриц
        M_list = list(torch.unbind(M, dim=1))  # каждый элемент shape (npnts,2,2)

        # Если нечётное число, добавим единичную матрицу (по npnts)
        I = torch.eye(2, dtype=M.dtype, device=device).unsqueeze(0).expand(npnts, -1, -1)
        while len(M_list) > 1:
            if len(M_list) % 2 == 1:
                M_list.append(I)
            M_list = [torch.matmul(M_list[i + 1], M_list[i]) for i in range(0, len(M_list), 2)]

        Mtot = M_list[0]  # shape (npnts,2,2)
        mrtot00 = Mtot[:, 0, 0]
        mrtot01 = Mtot[:, 0, 1]
        mrtot10 = Mtot[:, 1, 0]
        mrtot11 = Mtot[:, 1, 1]
    else:
        mrtot00 = torch.ones(npnts, dtype=torch.complex128, device=device)
        mrtot01 = rj[:, 0]

    r = mrtot01 / mrtot00
    reflectivity = r * torch.conj(r)

    # Расчет результатов
    r_real = r.real
    r_img = r.imag

    return reflectivity, r_real, r_img