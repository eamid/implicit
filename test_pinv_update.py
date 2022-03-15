from scipy.linalg import pinv
import numpy as np
from numpy.linalg import norm
# from scipy.linalg.blas import sgemm as mmmult
# from scipy.linalg.blas import sgemv as mvmult
from numpy import dot as mmmult
from numpy import dot as mvmult

def pinv_update(A, Ap, c, d, tol=1e-10):
    v = mvmult(Ap, c)
    n = mvmult(Ap.T, d)
    beta = 1.0 + np.dot(v, d)
    w = c - mvmult(A, v)
    m = d - mvmult(A.T, n)
    w_norm = norm(w) ** 2
    m_norm = norm(m) ** 2
    is_zero_w = np.sqrt(w_norm/len(w)) < tol
    is_zero_m = np.sqrt(m_norm/len(m)) < tol
    is_zero_beta = np.abs(beta) < tol
    if not is_zero_w and not is_zero_m:
        G = -np.outer(v, w)/w_norm - np.outer(m, n)/m_norm + beta * np.outer(m, w)/(m_norm * w_norm)
    elif is_zero_w and not is_zero_m and is_zero_beta:
        G = -np.outer(v, mvmult(Ap.T, v))/(norm(v) ** 2) - np.outer(m, n)/m_norm
    elif is_zero_w and not is_zero_beta:
        v_norm = norm(v) ** 2
        G = np.outer(m, mvmult(Ap.T, v))/beta - beta/(beta ** 2 + m_norm * v_norm) * np.outer(v_norm/beta * m + v, m_norm/beta * mvmult(Ap.T, v) + n)
    elif not is_zero_w and is_zero_m and is_zero_beta:
        G = -np.outer(mvmult(Ap, n), n)/(norm(n) ** 2) - np.outer(v, w)/w_norm
    elif is_zero_m and not is_zero_beta:
        n_norm = norm(n) ** 2
        G = np.outer(mvmult(Ap, n), w)/beta - beta/(beta ** 2 + w_norm * n_norm) * np.outer(w_norm/beta * mvmult(Ap, n) + v, n_norm/beta * w + n)
    elif is_zero_w and is_zero_m and is_zero_beta:
        v_norm = norm(v) ** 2
        n_norm = norm(n) ** 2
        G = -np.outer(v, mvmult(Ap.T, v))/v_norm - np.outer(mvmult(Ap, n, n))/n_norm + np.dot(v, mvmult(Ap, n))/(v_norm * n_norm) * np.outer(v, n)
    return Ap + G

def main():
    for itr in range(1000):
        C = np.random.random((50, 10))
        a = np.random.random((50))
        b = np.random.random((10))
        Cinv = pinv(C)
        Cinv_updated = pinv(C + np.outer(a, b))
        Cinv_output = pinv_update(C, Cinv, a, b)
        print(np.mean((Cinv_updated - Cinv_output)**2))

if __name__ == "__main__":
    main()
