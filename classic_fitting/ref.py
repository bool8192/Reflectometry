from calculating_dynamic_resolution import *
from convolution import convolution, convolution_dq
from variables import *
from read_complex_matrix import *
import torch.nn.functional as F
import torch

def smooth_filter(x, window_size=Ndots_extr, threshold=0.5):
    kernel = torch.ones(1, 1, window_size, dtype=x.dtype, device=x.device) / window_size

    pad_total = window_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad_left, pad_right), mode='replicate')
    smoothed = F.conv1d(x_padded, kernel).squeeze()

    if smoothed.shape[0] != x.shape[0]:
        smoothed = smoothed[:x.shape[0]]

    mask = torch.sigmoid(10 * (smoothed - threshold))
    return x * mask

def pseudoneuron(x):
    for i in range(5):
        x = smooth_filter(x, window_size=int(Ndots_extr/2))
    return x

def chain(r_conv):
    x = r_conv.real
    b = 3
    left = torch.roll(x, Ndots_extr)
    right = torch.roll(x, -Ndots_extr)

    x2 = torch.log(torch.pow((x - right) * (x - left), 2))
    left = torch.roll(x2, Ndots_extr)
    right = torch.roll(x2, -Ndots_extr)
    p_left = torch.sigmoid(b * (x2 - left))
    p_right = torch.sigmoid(b * (x2 - right))
    score = pseudoneuron(pow(p_left * p_right, 2))
    #score = pow(p_left * p_right, 1.2)
    return score

def reflectometry(q, matrix, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha2, var_delta_q):
    """
    Анализ спектра отражательной способности с поиском экстремумов
    
    Параметры:
    q: сетка волновых векторов
    matrix: матрица описания структуры
    
    Возвращает:
    #r: вектор всех значений коэффициента отражения
    #y: вектор всех значений коэффициента отражения со свёрткой
    #score: подсветка пиков в r_conv
    """
    r_1, r_real1, r_img1 = calculate_matrices_and_reflection(matrix, rough_res, kmax, q+var_delta_q, Ndots, gap)
    r_2, r_real2, r_img2 = calculate_matrices_and_reflection(matrix, rough_res, kmax, 2*(q+var_delta_q), Ndots, gap)

    r = (1-var_alpha2)*r_1 + var_alpha2*r_2
    
    r_conv = convolution_dq(r, kmax, q, Ndots_norm, var_sigma1, var_sigma2, gap) + var_Ibkg/var_I0

    #score = chain(r_conv)
    score = torch.zeros_like(r_conv, dtype=torch.float64)

    return r, r_conv, score


def reflectometry_freeform(q, matrix, var_sigma1, var_sigma2, var_I0, var_Ibkg, var_alpha2, var_delta_q):
    r_1, r_real1, r_img1 = calculate_matrices_and_reflection_freeform(matrix, rough_res, kmax, q+var_delta_q, Ndots, gap)
    r_2, r_real2, r_img2 = calculate_matrices_and_reflection_freeform(matrix, rough_res, kmax, 2*(q+var_delta_q), Ndots, gap)

    r = (1 - var_alpha2) * r_1 + var_alpha2 * r_2

    r_conv = convolution_dq(r, kmax, q, Ndots_norm, var_sigma1, var_sigma2, gap) + var_Ibkg / var_I0

    #score = chain(r_conv)
    score = torch.zeros_like(r_conv, dtype=torch.float64)

    return r, r_conv, score