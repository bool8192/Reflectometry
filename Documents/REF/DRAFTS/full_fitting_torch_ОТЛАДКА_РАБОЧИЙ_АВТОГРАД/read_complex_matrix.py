import torch
def read_complex_matrix(filename):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip('[] \n').split(',')
            row = [float(x.strip()) for x in values]
            matrix.append(row)
    pt_matrix = torch.tensor(matrix, dtype=torch.float64, device=device).requires_grad_(True)

    return pt_matrix