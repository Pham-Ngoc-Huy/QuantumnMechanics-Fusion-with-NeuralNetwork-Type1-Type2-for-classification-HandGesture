def matrix_multiply(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(A[0])  # the matching between A_rows and B_cols

    result = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            s = 0
            for k in range(inner):
                s += A[i][k] * B[k][j]
            result[i][j] = s

    return result
