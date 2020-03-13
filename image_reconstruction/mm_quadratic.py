import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import svds, bicg, LinearOperator

from optim_mdl import OptimModel


class MMQuadratic(OptimModel):
    def __init__(self, f, qmajor, grad_f, theta=1.9, max_iter=10, disable_progressbar=True):
        """

        :param f:
        :param qmajor: function of v, x such that qmajor(v, x) corresponds to
        the quadratic majorant in x valued in v
        :param grad_f:
        :param theta:
        :param max_iter:
        """
        super().__init__(f=f, grad_f=grad_f, disable_progressbar=disable_progressbar)
        self.qmajor = qmajor
        self.max_iter = max_iter
        self.theta = theta

        self.n = None

    def optimize(self, x0):
        x_n = x0.copy()
        self.n = len(x_n)

        self.init_time()
        for _ in tqdm(range(self.max_iter), disable=self.disable_progressbar):
            x_prev = x_n.copy()

            qmaj = self.get_operator(x_n)
            right_term = bicg(A=qmaj, b=self.grad_f(x_n))
            right_term = right_term[0]
            x_n = x_n - self.theta * right_term

            x_n = np.array(x_n)  # x_n was a matrix and not an array
            x_n = x_n.reshape(-1)
            self.f_eval(x_n)
            assert x_n.shape == x_prev.shape

            if self.has_converged(x_n):
                print('Method has converged')
                break

        return x_n, (self.times, self.f_vals)

    def get_operator(self, my_x):
        def matfunc(v):
            return self.qmajor(v, my_x)

        my_op = LinearOperator((self.n, self.n), matvec=matfunc, rmatvec=matfunc)
        return my_op

# class MMQuadratic(OptimModel):
#     def __init__(self, f, qmajor, grad_f, theta=1.9, max_iter=10):
#         super().__init__(f=f)
#         self.qmajor = qmajor
#         self.max_iter = max_iter
#         self.grad_f = grad_f
#         self.theta = theta
#
#     def optimize(self, x0):
#         x_n = x0.copy()
#         self.init_time()
#         for _ in tqdm(range(self.max_iter)):
#             x_prev = x_n.copy()
#             qmaj = self.qmajor(x_n)
#
#             # qmaj_inv = np.linalg.inv(qmaj)
#             # x_n = x_n - self.theta * qmaj_inv.dot(self.grad_f(x_n))
#
#             # TODO: Verify OK
#             right_term = np.linalg.solve(qmaj, self.grad_f(x_n))
#             x_n = x_n - right_term
#
#             x_n = np.array(x_n)  # x_n was a matrix and not an array
#             x_n = x_n.reshape(-1)
#             self.f_eval(x_n)
#
#             assert x_n.shape == x_prev.shape
#
#         return x_n, (self.times, self.f_vals)
