import numpy as np
def read_complex_matrix(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip('[] \n').split(',')
            row = [complex(x.strip()) for x in values]
            matrix.append(row)
    np_matrix = np.array(matrix)
    np_matrix[:, 0] *= 1e-10   
    np_matrix[:, 1] *= 1e+14 
    np_matrix[:, 2] *= 1e-10  
    return np_matrix