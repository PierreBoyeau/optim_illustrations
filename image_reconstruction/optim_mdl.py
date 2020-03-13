import numpy as np
import time


class OptimModel:
    def __init__(self, f, grad_f, disable_progressbar=True):
        self.f = f
        self.grad_f = grad_f
        self.times = []
        self.f_vals = []
        self.zero_time = None
        self.disable_progressbar = disable_progressbar

    def init_time(self):
        self.zero_time = time.time()

    def f_eval(self, x_val):
        assert self.zero_time is not None
        f_val = self.f(x_val)
        new_time = time.time() - self.zero_time

        self.times.append(new_time)
        self.f_vals.append(f_val)

    def has_converged(self, my_x):
        my_N = len(my_x)
        norm = np.linalg.norm(self.grad_f(my_x))
        return norm <= (np.sqrt(my_N) * 1e-4)
