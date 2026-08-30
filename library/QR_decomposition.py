import math

from .operator import subtract
from .scalar_multiply import projection, scalar_multiply
from .transpose import transpose


def QR_decomposition(matrix):
    Q = []
    R = [[0] * len(transpose(matrix)) for _ in range(len(transpose(matrix)))]  ## create matrix 0

    for idx, a in enumerate(transpose(matrix)):
        # case idx = 0 only
        if idx == 0:
            norm = math.sqrt(sum(c**2 for c in a))  # norm
            q = [c / norm for c in a]  # calculate q
            Q.append(q)
            R[idx][idx] = norm
        # other cases
        else:
            a_perpen = a[:]  # create a copy base on a
            for idx_q, q in enumerate(Q):
                dot_ab = scalar_multiply(a, q)
                dot_bb = scalar_multiply(q, q)
                coeff = dot_ab / dot_bb

                R[idx_q][idx] = coeff

                proj = projection(a, q)
                a_perpen = subtract(a_perpen, proj)

            norm = math.sqrt(sum(c**2 for c in a_perpen))
            R[idx][idx] = norm
            q = [c / norm for c in a_perpen]
            Q.append(q)
    return Q, R
