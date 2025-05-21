import torch
def read_complex_matrix(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip('[] \n').split(',')
            row = [complex(x.strip()) for x in values]
            matrix.append(row)
    pt_matrix = torch.tensor(matrix, dtype=torch.complex64)
    pt_matrix[:, 0] *= 1e-10   
    pt_matrix[:, 1] *= 1e+14 
    pt_matrix[:, 2] *= 1e-10  
    return pt_matrix