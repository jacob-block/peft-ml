import numpy as np


def mmt(A): return mat_stack_op(A, A)

def mat_stack_op(A, B): return np.matmul(A, B.transpose(0, 2, 1))

def mat_stack_ip(A, B): return np.matmul(A.transpose(0, 2, 1), B)

def mat_stack_op_sum(A, B): return np.sum(np.matmul(A, B.transpose(0, 2, 1)), axis=0)

def compute_A_loss(A_star, A, dim):
    return np.sum(np.square(A_star.reshape(dim, dim) - A))

def compute_U_loss(U_star, U):
    return np.sum(np.square(mat_stack_op(U_star, U_star) - mat_stack_op(U, U)))

def grad_func_gen(u_star: np.ndarray):
    num_tasks, dim, _ = u_star.shape
    u_star = u_star.reshape(num_tasks, dim, 1)
    us_ust = mat_stack_op(u_star, u_star)
    avg_us_ust = np.average(us_ust, axis=0).reshape(1, dim, dim)

    def flat_grad(u: np.ndarray):
        u = u.reshape(num_tasks, dim, 1)
        return ((mat_stack_op(u, u) - us_ust - mat_stack_op_sum(u, u) / num_tasks + avg_us_ust) @ u).ravel()

    return flat_grad


def extract_block_diag(a, n, k=0):
    # Extracts blocks of size n along kth diagonal
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")

    if k > 0:
        a = a[:, n * k:]
    else:
        a = a[-n * k:]

    n_blocks = min(a.shape[0] // n, a.shape[1] // n)

    new_shape = (n_blocks, n, n)
    new_strides = (n * a.strides[0] + n * a.strides[1],
                   a.strides[0], a.strides[1])

    return np.copy(np.lib.stride_tricks.as_strided(a, new_shape, new_strides))


def vec(mat):
    if mat.ndim != 2:
        raise ValueError("Only 2-D arrays handled")

    sh = mat.shape
    return np.concatenate(mat.T, axis=0).reshape(sh[0] * sh[1])
