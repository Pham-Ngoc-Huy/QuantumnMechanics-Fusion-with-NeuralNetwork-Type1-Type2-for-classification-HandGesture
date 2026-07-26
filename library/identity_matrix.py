def identity_matrix(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    if num_rows == num_cols:
        I = []
        for row in range(num_rows):
            current_row = []
            for col in range(num_cols):
                if col == row:
                    current_row.append(1)
                else:
                    current_row.append(0)
            I.append(current_row)
    return I