import numpy as np


def _proximal_power1(zeta, chi):
    return np.sign(zeta)*np.maximum(np.abs(zeta) - chi, np.zeros(zeta.shape))


def _proximal_power4thirds(zeta, chi):
    coef = (4.0*chi) / (3.0*(2.0**(1/3)))

    epsilon = np.sqrt(zeta**2 + (256.0*chi**3/729))
    term = ((epsilon - zeta)**(1/3)
            -(epsilon + zeta)**(1/3))
    return zeta + coef*term


def _proximal_power3halves(zeta, chi):
    coef = 9.0*chi*chi*np.sign(zeta)/8.0
    term = 1.0 - np.sqrt(1.0 + (16.0*np.abs(zeta))/(9.0*chi*chi))
    return zeta + coef*term


def _proximal_power2(zeta, chi):
    return zeta / (1.0+2.0*chi)


def _proximal_power3(zeta, chi):
    term = np.sqrt(1.0 + 12.0*chi*np.abs(zeta)) - 1.0
    term = term / (6*chi)
    return np.sign(zeta)*term


def _proximal_power4(zeta, chi):
    eps = np.sqrt(zeta*zeta + (1.0/(27*chi)))

    term1 = eps + zeta
    term1 = term1 / (8.0*chi)
    term2 = eps - zeta
    term2 = term2 / (8.0*chi)
    return (term1)**(1/3) - (term2)**(1/3)


class ProximalPower:
    def __init__(self, q, chi):
        self.power2fn = {
            1: _proximal_power1,
            4/3: _proximal_power4thirds,
            3/2: _proximal_power3halves,
            2: _proximal_power2,
            3: _proximal_power3,
            4: _proximal_power4
        }
        self.q = q
        self.chi = chi
        try:
            self.fn = self.power2fn[q]
        except KeyError:
            raise KeyError('Wrong choice of q: ', q)

    def compute(self, zeta):
        return self.fn(zeta, self.chi)
