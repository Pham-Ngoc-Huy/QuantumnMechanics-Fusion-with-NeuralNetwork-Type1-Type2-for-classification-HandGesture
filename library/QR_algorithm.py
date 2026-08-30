from .matrix_multiply import matrix_multiply
from .QR_decomposition import QR_decomposition
from .transpose import transpose


def QR_algorithm(A, max_iter=100):
    iter = 0
    while iter < max_iter:
        Q, R = QR_decomposition(A)
        Q_mat = transpose(Q)  # since this is vector -> need to tranpose
        A = matrix_multiply(R, Q_mat)
        iter += 1
    return A
