def scalar_multiply(vector_a, vector_b):
    result = 0
    for ele_1, ele_2 in zip(vector_a, vector_b):
        result += ele_1 * ele_2
    return result


def projection(vector_a, vector_b):
    dot_ab = scalar_multiply(vector_a=vector_a, vector_b=vector_b)
    dot_bb = scalar_multiply(vector_a=vector_b, vector_b=vector_b)

    coeff = dot_ab / dot_bb

    vector_scalar = [0 for _ in range(len(vector_b))]
    for idx in range(len(vector_b)):
        vector_scalar[idx] = coeff * vector_b[idx]
    return vector_scalar
