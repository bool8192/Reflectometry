import numpy as np
import math as ma

def transform_array(input_array, N, function):
    result = np.column_stack((input_array[:, 0], input_array[:, 1] - input_array[:, 2]))[:, :2].tolist()  #(ro, d)
    # Добавил промежуточные точки
    for i in range(1, len(input_array)):
        d2, ro2, rough2 = input_array[i]
        ro1 = input_array[i-1][1]
        insert_index = len(result) - (len(input_array) - i)
        for j in range(0, N+1):
            new_d = (rough2) / N
            new_ro =  ((ma.erf(-4*j/N+2)+1)/2)*(ro2-ro1)+ro1
            result.insert(insert_index, [new_d, new_ro])
    return np.array(result, dtype=complex)