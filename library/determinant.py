def get_minor(matrix, row, col):
    minor =[]
    for i in range(len(matrix)): # this is for row
        if i == row:
            continue
        new_row = []
        for j in range(len(matrix)): # this is for column
            if j == col:
                continue
            new_row.append(matrix[i][j])
        minor.append(new_row)
    return minor

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] + matrix[0][1]*matrix[1][0]
    
    det = 0
    for j in range(n): # this is for row calculation
        sign = (-1) ** j # (-1)^n -> odd: negative, even: positive
        minor = get_minor(matrix, 0, j)
        print(minor)
        det += sign * matrix[0][j] * determinant(minor)
        print(det)
    
    return det