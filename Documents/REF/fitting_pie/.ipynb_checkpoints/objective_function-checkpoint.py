from ref import reflectometry
from variables import sigma
import torch
import numpy as np

def repair_matrix(bounds, vector):
       mask = bounds[..., 0] != bounds[..., 1]
       flat_mask = mask.flatten()
       indices_flat = flat_mask.nonzero().squeeze()
       flat_base = bounds[..., 0].flatten()
       result = flat_base.scatter(0, indices_flat, vector)
       target_shape = bounds[..., 0].shape
       viewed_result = result.view_as(bounds[..., 0])
       return viewed_result

class Comparator:
    def __init__(self, q, orig_matr, orig_sigma1, orig_sigma2, orig_I0, orig_Ibkg):
        self.q = q
        self.orig_matr = orig_matr
        self.orig_sigma1 = orig_sigma1
        self.orig_sigma2 = orig_sigma2
        self.orig_I0 = orig_I0
        self.orig_Ibkg = orig_Ibkg
        
    def compare(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        #print("I0 inside comparator, grad_fn:", var_I0.grad_fn)
        #print("Ibkg inside comparator, grad_fn:", var_Ibkg.grad_fn)
        #print("sigma inside comparator, grad_fn:", var_sigma1.grad_fn)
        r1, r_conv1 = reflectometry(self.q, self.orig_matr, self.orig_sigma1, self.orig_sigma2, self.orig_I0, self.orig_Ibkg)
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)
        
        diff = r_conv1 - r_conv2
        squared = diff ** 2
        loss = torch.sum(squared)
    
        return loss

    def compare_weightless(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        r1, r_conv1 = reflectometry(self.q, self.orig_matr, self.orig_sigma1, self.orig_sigma2, self.orig_I0, self.orig_Ibkg)
        r2, r_conv2 = reflectometry(self.q, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg)
    
        ratio = (r_conv1 - r_conv2)/r_conv1
        squared = ratio ** 2
        loss = torch.sum(squared)
    
        return loss



class varbounds:
    def __init__(self, bounds_tensor, loss_func):
        self.loss = loss_func
        self.bounds = bounds_tensor
        self.mask = self.bounds[..., 0] != self.bounds[..., 1]
        flat_mask = self.mask.flatten()
        self.indices_flat = flat_mask.nonzero().squeeze()
        
    def objective_function(self, var_vector_d, var_vector_rho, var_sigma1, var_sigma2, var_I0, var_Ibkg):
        var_vector = torch.cat((var_vector_d.reshape(-1, 2), var_vector_rho.reshape(-1, 1)), dim=1)[:,torch.tensor([0, 2, 1])].flatten()
        
        flat_base = self.bounds[..., 0].flatten()
        result = flat_base.scatter(0, self.indices_flat, var_vector)
        target_shape = self.bounds[..., 0].shape
        viewed_result = result.view_as(self.bounds[..., 0])
        final_result = self.loss(viewed_result, var_sigma1, var_sigma2, var_I0, var_Ibkg)

        return final_result



class varsigma:
    def __init__(self, d_vector, rho_vector, I, Ibkg, loss_func, all_bounds):
        self.I = I
        self.Ibkg = Ibkg
        self.loss = loss_func
        self.matrix = repair_matrix(all_bounds, torch.cat((d_vector.reshape(-1, 2), rho_vector.reshape(-1, 1)), dim = 1)[:, [0, 2, 1]].flatten())

    def objective_function(self, sigmas):
         return self.loss(self.matrix, sigmas[0], sigmas[1], self.I, self.Ibkg).detach().numpy()
    