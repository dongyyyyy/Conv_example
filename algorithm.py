'''
import numpy as np


def find_max_matrix(i,j,matrix):
    if i == matrix.shape[0]-1 and j == matrix.shape[1]-1:
        return matrix[i][j]
    elif i == matrix.shape[0]-1 and j < matrix.shape[1]-1:
        return find_max_matrix(i,j+1,matrix) + matrix[i][j]
    elif i < matrix.shape[0]-1 and j == matrix.shape[1]-1:
        return find_max_matrix(i+1,j,matrix) + matrix[i][j]
    else:
        a = find_max_matrix(i+1,j,matrix) + matrix[i][j]
        b = find_max_matrix(i, j+1, matrix) + + matrix[i][j]
        if a > b:
            return a
        else:
            return b


max_value = 100
matrix_shape = (4,4)
a = np.random.randint(max_value,size=matrix_shape)

print(a)

c = find_max_matrix(0,0,a)
print(c)
'''
