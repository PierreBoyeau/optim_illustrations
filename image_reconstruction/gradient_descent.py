import numpy as np
from tqdm import tqdm

from optim_mdl import OptimModel


class GradientDescent(OptimModel):
    def __init__(self, f, nu, grad_f, x_dim, max_iter, disable_progressbar=True):
        super().__init__(f=f, grad_f=grad_f, disable_progressbar=disable_progressbar)

        self.nu = nu
        self.gamma = 1.9 / nu
        self.x_dim = x_dim
        self.max_iter = max_iter

    def optimize(self, x0):
        x_n = x0.copy()
        self.init_time()
        for _ in tqdm(range(self.max_iter), disable=self.disable_progressbar):
            x_prev = x_n.copy()
            x_n = x_n - self.gamma * self.grad_f(x_n)

            self.f_eval(x_n)

            if self.has_converged(x_n):
                print('Method has converged')
                break
        return x_n, (self.times, self.f_vals)


