from add_roughness import transform_array
from variables import rough_res
from ref import reflectometry
import matplotlib.pyplot as plt
from variables import sigma
import torch
import numpy as np

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
        
    def compare(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r_conv1 = self.r
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)
        
        diff = r_conv1 - r_conv2
        squared = diff ** 2
        loss = torch.sum(squared)
    
        return loss

    def compare_weightless(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r_conv1 = self.r
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        ratio = (r_conv1 - r_conv2)/r_conv1
        squared = ratio ** 2
        loss = torch.sum(squared)
    
        return loss

    def compare_08(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r_conv1 = self.r
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_conv1, 0.6)
        squared = ratio ** 2
        loss = torch.sum(squared)

        return loss

    def compare_05(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r_conv1 = self.r
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_conv1, 0.75)
        squared = ratio ** 2
        loss = torch.sum(squared)

        return loss

    def compare_025(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r_conv1 = self.r
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        ratio = (r_conv1 - r_conv2) / torch.pow(r_conv1, 0.875)
        squared = ratio ** 2
        loss = torch.sum(squared)

        return loss



class varbounds:
    def __init__(self, bounds_tensor, loss_func, norm_coef, matrixx, sigma1, sigma2, I0, Ibkg):
        self.loss = loss_func
        self.bounds = bounds_tensor
        self.mask = self.bounds[..., 0] != self.bounds[..., 1]
        flat_mask = self.mask.flatten()
        self.indices_flat = flat_mask.nonzero().squeeze()
        matrix = transform_array(matrixx, rough_res)
        with torch.no_grad():
            self.coef_normalization = norm_coef*loss_func(matrixx, sigma1, sigma2, I0, Ibkg)/torch.sum(torch.abs((matrix[0:-2, 1] + matrix[2:, 1] - 2 * matrix[1:-1, 1]) / pow((matrix[0:-2, 0] + matrix[2:, 0]), 2)))

        
    def objective_function(self, var_vector_d, var_vector_rho, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        var_vector = torch.cat((var_vector_d.reshape(-1, 8), var_vector_rho.reshape(-1, 1)), dim=1)[:,torch.tensor([0, 8, 1, 2, 3, 4, 5, 6, 7])].flatten()
        
        flat_base0 = self.bounds[..., 0].flatten()
        flat_base = torch.complex(flat_base0, torch.zeros_like(flat_base0))
        result = flat_base.scatter(0, self.indices_flat, var_vector)
        viewed_result = result.view_as(self.bounds[..., 0])

        final_result = self.loss(viewed_result, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        #matrix = transform_array(viewed_result, rough_res)
        normalization = 0
        #= self.coef_normalization * torch.sum(torch.abs((matrix[0:-2, 1] + matrix[2:, 1] - 2 * matrix[1:-1, 1]) / pow((matrix[0:-2, 0] + matrix[2:, 0]), 2))

        final_result = final_result + normalization

        return final_result



class varsigma:
    def __init__(self, d_vector, r_rho_vector, i_rho_vector, I, Ibkg, loss_func, all_bounds):
        self.I = I
        self.Ibkg = Ibkg
        self.loss = loss_func
        self.matrix = repair_matrix(all_bounds, torch.cat((d_vector.reshape(-1, 8), torch.complex(r_rho_vector, i_rho_vector).reshape(-1, 1)), dim = 1)[:, [0, 8, 1, 2, 3, 4, 5, 6, 7]].flatten())

    def objective_function(self, sigmas):
         return self.loss(self.matrix, sigmas[0], sigmas[1], self.I, self.Ibkg).cpu().detach().numpy()
    