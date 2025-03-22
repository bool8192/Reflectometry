import numpy as np
def read_complex_matrix(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip('[] \n').split(',')
            row = [complex(x.strip()) for x in values]
            matrix.append(row)
    return np.array(matrix)