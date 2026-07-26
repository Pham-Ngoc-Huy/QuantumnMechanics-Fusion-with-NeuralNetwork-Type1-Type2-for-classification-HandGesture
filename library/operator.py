def subtract(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have same length")
    return [a - b for a, b in zip(vector_a, vector_b)]
def add(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have same length")
    return [a + b for a, b in zip(vector_a, vector_b)]