import numpy as np
from utilities import mat_stack_op, mmt
rng = np.random.default_rng()

def arr_tp(arr): return arr.transpose(0, 2, 1)

def fro_sq(X): return np.sum(np.square(X))

def ith_stack_op(A, B, i): return A[i] @ B[i].transpose()

class MetaLearner:
    def __init__(self, dim, num_samples, low_rank_dim, num_tasks, tasks, sigma=0.1, inner_learning_rate=0.01,
                 outer_learning_rate=0.01, data=None, symmetric=True):
        self.dim = dim
        self.num_samples = num_samples
        self.low_rank_dim = low_rank_dim
        self.num_tasks = num_tasks
        self.sigma = sigma
        self.tasks = tasks
        self.outer_learning_rate = outer_learning_rate
        self.inner_learning_rate = inner_learning_rate
        self.symmetric = symmetric
        if data is None:
            self.gen_data()
        else:
            self.X, self.Y = data
        self.A = np.zeros((self.dim, self.dim))
        self.U = rng.normal(0, 0.1, (self.num_tasks, self.dim, self.low_rank_dim))
        self.V = rng.normal(0, 0.1, (self.num_tasks, self.dim, self.low_rank_dim))
        self.XXT = self.X @ arr_tp(self.X)

    def gen_data(self):
        noise = rng.normal(0, self.sigma ** 2, (self.num_tasks, self.dim, self.num_samples))
        self.X = rng.normal(0, 1, (self.num_tasks, self.dim, self.num_samples))
        self.Y = self.tasks @ self.X + noise

    def inner_gradient_step(self):
        if self.symmetric:
            self.U -= self.inner_learning_rate * self.inner_gradient_symmetric() / self.num_samples
        else:
            u_grad, v_grad = self.inner_gradient_asymmetric()
            self.U -= self.inner_learning_rate * u_grad / self.num_samples
            self.V -= self.inner_learning_rate * v_grad / self.num_samples

    def outer_gradient_step(self):
        self.A -= self.outer_learning_rate * self.outer_gradient() / (self.num_samples * self.num_tasks)

    def inner_gradient_symmetric(self):
        y_prime = (self.A + mat_stack_op(self.U, self.U)) @ self.X - self.Y
        return (self.X @ arr_tp(y_prime) + y_prime @ arr_tp(self.X)) @ self.U

    def inner_gradient_asymmetric(self):
        y_prime = (self.A + mat_stack_op(self.U, self.V)) @ self.X - self.Y
        return y_prime @ arr_tp(self.X) @ self.V, self.X @ arr_tp(y_prime) @ self.U

    def outer_gradient(self):
        return (((self.A + mat_stack_op(self.U, self.U)) @ self.X - self.Y) @ arr_tp(self.X)).sum(axis=0)

    def loss(self):
        if self.symmetric:
            return np.sum(((self.Y - (self.A + mmt(self.U)) @ self.X) ** 2)) / self.num_tasks
        else:
            return np.sum(((self.Y - (self.A + mat_stack_op(self.U, self.V)) @ self.X) ** 2)) / self.num_tasks

    def semi_true_loss(self):
        if self.symmetric:
            return np.sum(((self.tasks @ self.X - (self.A + mmt(self.U)) @ self.X) ** 2)) / self.num_tasks
        else:
            return np.sum(
                ((self.tasks @ self.X - (self.A + mat_stack_op(self.U, self.V)) @ self.X) ** 2)) / self.num_tasks

    def true_loss(self):
        return np.sum((self.tasks - self.A - mat_stack_op(self.U, self.U if self.symmetric else self.V)) ** 2) / 2

    def detach(self, tasks, num_samples, low_rank_dim_mult, new_num_tasks=1, data=None, new_learning_rate=None):
        if new_learning_rate is None:
            new_learning_rate = self.inner_learning_rate
        newLearner = MetaLearner(
            self.dim,
            num_samples,
            self.low_rank_dim * low_rank_dim_mult,
            new_num_tasks,
            tasks,
            # 0,
            self.sigma,
            new_learning_rate,
            self.outer_learning_rate,
            data,
            symmetric=False
        )
        newLearner.A = self.A
        return newLearner
