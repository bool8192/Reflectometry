from ref import reflectometry
from variables import sigma
import torch


class Comparator:
    def __init__(self, q, orig_matr, orig_sigma1, orig_sigma2, orig_I0, orig_Ibkg):
        self.q = q
        self.orig_matr = orig_matr
        self.orig_sigma1 = orig_sigma1
        self.orig_sigma2 = orig_sigma2
        self.orig_I0 = orig_I0
        self.orig_Ibkg = orig_Ibkg
        
    def compare(self, var_matr, var_sigma1, var_sigma2, var_I0, var_Ibkg):
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