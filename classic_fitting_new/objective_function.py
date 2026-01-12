from add_roughness import transform_array
from variables import rough_res
from ref import *
import matplotlib.pyplot as plt
from variables import sigma
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def repair_matrix(bounds, vector):
    flat_mask = (bounds[..., 0] != bounds[..., 1])

    indices_flat = flat_mask.flatten().nonzero(as_tuple=True)[0]
    flat_base0 = bounds[..., 0].flatten().to(torch.float64)
    flat_base = torch.complex(flat_base0, torch.zeros_like(flat_base0))
    vector = vector.to(dtype=flat_base.dtype)

    result = flat_base.scatter(0, indices_flat, vector)
    viewed_result = result.view_as(bounds[..., 0])
    return viewed_result


class Comparator:
    def __init__(self, q, r):
        self.q = q
        self.r = r

    def compare(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q, ref_type='model'):
        r_conv1 = self.r
        score1 = chain(r_conv1)
        if ref_type == 'model':
            r2, r_conv2, score2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)
        else:
            r2, r_conv2, score2 = reflectometry_freeform(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)

        diff = r_conv1 - r_conv2
        squared = diff ** 2
        loss = torch.sum(squared).real + torch.sum(torch.pow(score1-score2, 2))
        #print(torch.sum(squared).real,torch.sum(torch.pow(score1-score2, 2)))

        return loss

    def compare_weightless(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q, ref_type='model'):
        r_conv1 = self.r
        score1 = chain(r_conv1)
        if ref_type == 'model':
            r2, r_conv2, score2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)
        else:
            r2, r_conv2, score2 = reflectometry_freeform(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)

        r_min = torch.minimum(r_conv1.real, r_conv2.real)

        ratio = (r_conv1 - r_conv2) / r_min
        squared = ratio ** 2
        loss = torch.sum(squared).real + torch.sum(torch.pow(score1-score2, 2))
        #print(torch.sum(squared).real , torch.sum(torch.pow(score1 - score2, 2)))

        return loss

    def compare_08(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q, ref_type='model'):
        r_conv1 = self.r
        score1 = chain(r_conv1)
        if ref_type == 'model':
            r2, r_conv2, score2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)
        else:
            r2, r_conv2, score2 = reflectometry_freeform(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)

        r_min = torch.minimum(r_conv1.real, r_conv2.real)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_min, 0.6)
        squared = ratio ** 2
        loss = torch.sum(squared).real + torch.sum(torch.pow(score1-score2, 2))
        #print(torch.sum(squared).real , torch.sum(torch.pow(score1 - score2, 2)))

        return loss

    def compare_05(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q, ref_type='model'):
        r_conv1 = self.r
        score1 = chain(r_conv1)
        if ref_type == 'model':
            r2, r_conv2, score2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)
        else:
            r2, r_conv2, score2 = reflectometry_freeform(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)

        r_min = torch.minimum(r_conv1.real, r_conv2.real)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_min, 0.75)
        squared = ratio ** 2
        loss = torch.sum(squared).real + torch.sum(torch.pow(score1-score2, 2))
        #print(torch.sum(squared).real , torch.sum(torch.pow(score1 - score2, 2)))

        return loss

    def compare_025(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q, ref_type='model'):
        r_conv1 = self.r
        score1 = chain(r_conv1)
        if ref_type == 'model':
            r2, r_conv2, score2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)
        else:
            r2, r_conv2, score2 = reflectometry_freeform(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha, var_delta_q)

        r_min = torch.minimum(r_conv1.real, r_conv2.real)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_min, 0.875)
        squared = ratio ** 2
        loss = torch.sum(squared).real + torch.sum(torch.pow(score1-score2, 2))
        #print(torch.sum(squared).real, torch.sum(torch.pow(score1 - score2, 2)))

        return loss



class varbounds:
    def __init__(self, bounds_tensor, loss_func, norm_coef, matrixx, sigma1, sigma2, I0, Ibkg, alpha2, delta_q, ref_type='model'):
        self.ref_type = ref_type
        self.loss = loss_func
        self.bounds = bounds_tensor
        self.mask = self.bounds[..., 0] != self.bounds[..., 1]
        flat_mask = self.mask.flatten()
        self.indices_flat = flat_mask.nonzero().squeeze()
        if self.ref_type == 'model':
            matrix = transform_array(matrixx, rough_res)
            with torch.no_grad():
                self.coef_normalization = norm_coef * loss_func(matrixx, sigma1, sigma2, I0, Ibkg, alpha2,  delta_q, ref_type = ref_type) / torch.sum(torch.abs((matrix[0:-2, 1] + matrix[2:, 1] - 2 * matrix[1:-1, 1]) / pow((matrix[0:-2, 0] + matrix[2:, 0]), 2)))
        else:
            matrix = matrixx
            with torch.no_grad():
                self.coef_normalization = norm_coef*loss_func(matrix, sigma1, sigma2, I0, Ibkg, alpha2, delta_q, ref_type = ref_type)/torch.sum(torch.abs((matrix[0:-2, 1] + matrix[2:, 1] - 2 * matrix[1:-1, 1]) / pow((matrix[0:-2, 0] + matrix[2:, 0]), 2)))

        
    def objective_function(self, var_vector_d, var_vector_rho, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha2, var_delta_q):
        if self.ref_type == 'model':
            var_vector = torch.cat((var_vector_d.reshape(-1, 5), var_vector_rho.reshape(-1, 1)), dim=1)[:,
                         torch.tensor([0, 5, 1, 2, 3, 4])].flatten()
            flat_base0 = self.bounds[..., 0].flatten()
            flat_base = torch.complex(flat_base0, torch.zeros_like(flat_base0))
            result = flat_base.scatter(0, self.indices_flat, var_vector)
            viewed_result = result.view_as(self.bounds[..., 0])
            normalization = 0
        else:
            var_vector = torch.cat((var_vector_d.reshape(-1, 2), var_vector_rho.reshape(-1, 1)), dim=1)[:,torch.tensor([0, 2, 1])]
            viewed_result = var_vector
            normalization = self.coef_normalization * torch.sum(torch.abs((viewed_result[0:-2, 1] + viewed_result[2:, 1] - 2 * viewed_result[1:-1, 1]) / pow((viewed_result[0:-2, 0] + viewed_result[2:, 0]), 2)))

        final_result = self.loss(viewed_result, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha2, var_delta_q, self.ref_type)

        final_result = final_result + normalization

        return final_result



class varsigma:
    def __init__(self, d_vector, r_rho_vector, i_rho_vector, I, Ibkg, alpha2, delta_q, loss_func, all_bounds):
        self.I = I
        self.Ibkg = Ibkg
        self.loss = loss_func
        self.alpha2 = alpha2
        self.delta_q = delta_q
        self.matrix = repair_matrix(all_bounds, torch.cat((d_vector.reshape(-1, 5), torch.complex(r_rho_vector, i_rho_vector).reshape(-1, 1)), dim = 1)[:, [0, 5, 1, 2, 3, 4]].flatten())

    def objective_function(self, sigmas):
         return self.loss(self.matrix, sigmas[0], sigmas[1], self.I, self.Ibkg, self.alpha2, self.delta_q).cpu().detach().numpy()


conv = nn.Conv1d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=1
)

def gaussian_smooth_1d(x1, kernel_size=7, sigma=None):
    """
    x — 1D тензор [L]
    kernel_size — число точек (нечётное)
    sigma — если None, ставим kernel_size / 6
    """

    x = torch.tensor(x1)  # <---- приведение типа

    if sigma is None:
        sigma = kernel_size / 6.0

    half = kernel_size // 2
    t = torch.linspace(-half, half, steps=kernel_size, dtype=x.dtype)

    kernel = torch.exp(-0.5 * (t / sigma)**2)
    kernel /= kernel.sum()

    kernel = kernel.view(1, 1, -1).to(x.dtype)

    x = x.view(1, 1, -1)

    y = F.conv1d(x, kernel, padding=half)

    return y.view(-1)


def period_loss(q, y, rx):
    q = torch.tensor(q, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    rx = torch.tensor(rx, dtype=torch.float)

    # --- поиск первой точки < 0.5 ---
    below_threshold = torch.where(rx < 0.2)[0]
    below_threshold1 = torch.where(y < 0.2)[0]

    crit_dot = below_threshold[0] if len(below_threshold) else 0
    crit_dot1 = below_threshold1[0] if len(below_threshold1) else 0

    # --- подготовка соседних точек ---
    left, center, right = y[:-2], y[1:-1], y[2:]
    left1, center1, right1 = rx[:-2], rx[1:-1], rx[2:]

    # --- маски экстремумов ---
    max_mask = (center > left) & (center > right)
    min_mask = (center < left) & (center < right)
    max_mask1 = (center1 > left1) & (center1 > right1)
    min_mask1 = (center1 < left1) & (center1 < right1)

    # --- индексы экстремумов ---
    raw_max_indices = torch.where(max_mask)[0] + 1
    raw_min_indices = torch.where(min_mask)[0] + 1
    raw_max_indices1 = torch.where(max_mask1)[0] + 1
    raw_min_indices1 = torch.where(min_mask1)[0] + 1

    max_indices = raw_max_indices[raw_max_indices > crit_dot]
    min_indices = raw_min_indices[raw_min_indices > crit_dot]
    max_indices1 = raw_max_indices1[raw_max_indices1 > crit_dot1]
    min_indices1 = raw_min_indices1[raw_min_indices1 > crit_dot1]

    max_indices = max_indices[4:26]
    min_indices = min_indices[4:26]
    max_indices1 = max_indices1[4:27]
    min_indices1 = min_indices1[4:27]

    # --- координаты экстремумов ---
    x_max, x_min = q[max_indices], q[min_indices]
    x_max1, x_min1 = q[max_indices1], q[min_indices1]

    # --- расчёт периода ---
    avg_period_mod = (torch.mean(x_max[1:] - x_max[:-1]) +
                      torch.mean(x_min[1:] - x_min[:-1])) * 0.5e-10
    avg_period_exp = (torch.mean(x_max1[1:] - x_max1[:-1]) +
                      torch.mean(x_min1[1:] - x_min1[:-1])) * 0.5e-10
    if torch.isnan(avg_period_mod).any(): return 1.0

    return float(abs(avg_period_mod - avg_period_exp) / torch.min(avg_period_exp, avg_period_mod))


def reflectometry_trace(q, y):
    """
    В коде функции обращаемся к параметрам как глобальным переменным
    Анализ спектра отражательной способности с поиском экстремумов

    Параметры:
    kmax: максимальное значение волнового вектора
    gap: относительная длина перехода между разрешениями
    Ndots: количество точек сетки по дефолту
    Ndots_norm: количество точек в нормальном распределение свёртки
    sigma1, sigma2: разрешение установки
    rough_res: число переходных слоёв шероховатости

    Возвращает:
    x_crit: критическая точка
    x_max/y_max: координаты максимумов
    x_min/y_min: координаты минимумов
    ----
    q: вектор всех значений волнового вектора
    y: вектор всех значений коэффициента отражения со свёрткой
    """

    q = torch.tensor(q, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    # Поиск критической точки
    below_threshold = torch.where(y < 0.2)[0]
    if len(below_threshold) == 0:
        crit_dot = 0
    else:
        crit_dot = below_threshold[0]

    # Поиск локальных экстремумов
    left = y[:-2]
    center = y[1:-1]
    right = y[2:]

    max_mask = (center > left) & (center > right)
    min_mask = (center < left) & (center < right)

    # Фильтрация индексов
    raw_max_indices = torch.where(max_mask)[0] + 1
    raw_min_indices = torch.where(min_mask)[0] + 1

    max_indices = raw_max_indices[raw_max_indices > crit_dot]
    min_indices = raw_min_indices[raw_min_indices > crit_dot]

    max_indices = max_indices[4:26]
    min_indices = min_indices[4:26]

    # Получение координат и их
    x_max = (q[max_indices])[0:Ndots_trace]
    y_max = (y[max_indices])[0:Ndots_trace]
    x_min = (q[min_indices])[0:Ndots_trace]
    y_min = (y[min_indices])[0:Ndots_trace]
    #x_crit = q[crit_dot]

    return x_max, y_max, x_min, y_min

def trace_loss(q, y, rx):
    #y = gaussian_smooth_1d(y)
    y_trace1, y_trace2, y_trace3, y_trace4 = reflectometry_trace(q, y)
    rx_trace1, rx_trace2, rx_trace3, rx_trace4 = reflectometry_trace(q, rx)
    loss = torch.sum(torch.abs((rx_trace1 - y_trace1)/y_trace1)) + torch.sum(torch.abs((rx_trace2 - y_trace2)/y_trace2)) + torch.sum(torch.abs((rx_trace3 - y_trace3)/y_trace3)) + torch.sum(torch.abs((rx_trace4 - y_trace4)/y_trace4))
    return loss