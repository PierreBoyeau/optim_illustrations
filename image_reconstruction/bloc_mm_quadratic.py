from tqdm import tqdm
import numpy as np
from scipy.sparse.linalg import svds, bicg, LinearOperator
from scipy.sparse import diags

from optim_mdl import OptimModel


class BlockMM(OptimModel):
    def __init__(self, f, n_j, grad_f, get_aj, n, max_iter=10, theta=1.9,
                 disable_progressbar=True):
        """
        As a first approximation I consider that computing all gradient elements
        Even if we only care about the j indices have more or less the same cost
        Because it will always be minor compared to the matrix inversion
        we have to perform

        If poor performance, should compute specifically needed quantity

        :param f:
        :param n_j:
        :param grad_f:
        :param get_aj:
        :param n:
        :param max_iter:
        :param theta:
        """

        super().__init__(f=f, grad_f=grad_f, disable_progressbar=disable_progressbar)
        self.n_j = n_j
        self.n = n
        self.get_a_j = get_aj
        assert n % n_j == 0
        self.j_total = int(n / n_j)
        self.max_iter = max_iter
        self.theta = theta

    def optimize(self, x0):
        x_n = x0.copy()
        self.init_time()

        for k in tqdm(range(self.max_iter), disable=self.disable_progressbar):
            j = (k-1) % self.j_total
            j += 1
            block_indices = self.get_block_indices(j)
            all_grad = self.grad_f(x_n)
            grad_j = all_grad[block_indices]

            # a_j = self.get_a_j(my_x=x_n, block_indices=block_indices)
            # x_n_j = x_n[block_indices].copy()
            # a_j_inv = np.linalg.inv(a_j)
            # x_n_j = x_n_j - self.theta * a_j_inv.dot(grad_j)

            a_j = self.get_operator(self.get_a_j(my_x=x_n, block_indices=block_indices))
            # print(a_j.shape, grad_j.shape)
            right_term = bicg(A=a_j, b=grad_j)
            right_term = right_term[0]
            x_n_j = x_n[block_indices].copy()
            x_n_j = x_n_j - self.theta * right_term

            x_n[block_indices] = x_n_j
            self.f_eval(x_n)

            if self.has_converged(x_n):
                print('Method has converged')
                break

        return x_n, (self.times, self.f_vals)

    def get_block_indices(self, j):
        """

        :param j:
        :return:
        """
        assert j > 0
        res = np.arange(self.n_j * (j - 1), self.n_j * j)
        assert len(res) == self.n_j
        return res

    def get_operator(self, func):
        my_op = LinearOperator((self.n_j, self.n_j), matvec=func, rmatvec=func)
        return my_op
