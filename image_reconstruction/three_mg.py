import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from optim_mdl import OptimModel


class ThreeMG(OptimModel):
    def __init__(self, f, my_h, my_g, my_delta, my_lambda, grad_f, max_iter=50,
                 disable_progressbar=True):
        super().__init__(f=f, grad_f=grad_f, disable_progressbar=disable_progressbar)
        self.my_h = my_h
        self.my_g = my_g
        self.my_delta = my_delta
        self.my_lambda = my_lambda

        self.max_iter = max_iter

        self.current_x = None
        self.previous_x = None
        self.dk = None
        self.uk = None

        m, n = self.my_h.shape
        self.m = m
        self.n = n

    def optimize(self, x0):
        x_n = x0.copy()
        self.init_time()

        for _ in tqdm(range(self.max_iter), disable=self.disable_progressbar):
            self.current_x = x_n.copy()

            self.dk_compute()
            self.uk_compute()
            # TODO: Verify dimensions
            new_contrib = self.dk.dot(self.uk)
            assert new_contrib.shape == x_n.shape
            x_n = x_n + new_contrib
            self.previous_x = self.current_x.copy()

            self.f_eval(x_n)

            if self.has_converged(x_n):
                print('Method has converged')
                break

        return x_n, (self.times, self.f_vals)

    def dk_compute(self):
        if self.dk is None:
            self.dk = -self.grad_f(self.current_x)
            self.dk = self.dk.reshape((-1, 1))
        else:
            assert self.previous_x is not None
            dk = np.zeros((self.n, 2))

            delta_x = self.current_x - self.previous_x
            grad_x = self.grad_f(self.current_x)

            assert delta_x.shape == (self.n,)
            assert grad_x.shape == (self.n,)

            dk[:, 0] = delta_x
            dk[:, 1] = grad_x

            self.dk = dk

            assert self.dk.shape == (self.n, 2)

    def uk_compute(self):
        """
        Recall that for our very specific example the
        hessian is:
        H^TH + \lambda + G^T D diag G
        :return:
        """
        hd_term = self.my_h.dot(self.dk)
        first_term = hd_term.T.dot(hd_term)

        gd_term = self.my_g.dot(self.dk)
        gx = self.my_g.dot(self.current_x)

        # psi_ddot = self.my_delta * (self.my_delta ** 2 + gx ** 2) ** (-1.5)
        diag_term = 1.0 / (self.my_delta * np.sqrt(gx ** 2 + self.my_delta ** 2))
        mat_diag_term = diags(diag_term)

        second_term = gd_term.T.dot(mat_diag_term.dot(gd_term))

        prefactor = first_term + self.my_lambda * second_term
        prefactor = np.linalg.pinv(prefactor)

        last_term = self.dk.T.dot(self.grad_f(self.current_x))

        self.uk = -prefactor.dot(last_term)