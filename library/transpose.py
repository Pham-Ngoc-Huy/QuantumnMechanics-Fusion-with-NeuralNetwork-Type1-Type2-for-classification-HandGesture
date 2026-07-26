def transpose(matrix):
    matrix_transpose = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix_transpose[col][row] = matrix[row][col]
    return matrix_transpose