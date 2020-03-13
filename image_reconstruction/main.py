
from gradient_descent import GradientDescent
from mm_quadratic import MMQuadratic
from bloc_mm_quadratic import BlockMM
from three_mg import ThreeMG


from scipy.sparse.linalg import svds, bicg, LinearOperator
from scipy.sparse import diags
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

SIGMA = 1.0

h = loadmat('/home/pierre/MVA/distrib_optim/TP4/data/H.mat')['H']
x = loadmat('/home/pierre/MVA/distrib_optim/TP4/data/x.mat')['x']
g = loadmat('/home/pierre/MVA/distrib_optim/TP4/data/G.mat')['G']

y = h.dot(x) + (SIGMA*np.random.randn(16200, 1))
y = y.reshape(-1)

N = 90*90
M = 90*180

lamb, delta = 0.13, 2e-2


def f(my_x):
    first_vec = h.dot(my_x) - y
    first_vec = first_vec ** 2
    assert first_vec.shape == (16200,), first_vec.shape
    first_scal = 0.5 * first_vec.sum()

    g_x = g.dot(my_x)
    assert g_x.shape == (16200,)
    second_vec = np.sqrt(1.0 + (g_x ** 2) / (delta ** 2))
    second_scal = lamb * second_vec.sum()

    res = first_scal - second_scal
    return res


x0 = np.zeros(N)
h_transpose_h = h.T.dot(h)


def grad_f(my_x):
    first_prod = h.dot(my_x) - y
    first_term = h.T.dot(first_prod)

    gx = g.dot(my_x)
    dot_phi = gx / (delta * np.sqrt(gx ** 2 + delta ** 2))

    grad = first_term + lamb * g.T.dot(dot_phi)
    return grad


def single_plot(stats, label=None):
    times, vals = stats
    plt.plot(times, vals, label=label)
    plt.xscale('log')


if __name__ == '__main__':
    ##############################
    ##############################

    def a_major_func(my_v, my_x):
        hess_v = h.T.dot(h.dot(my_v))
        gx = g.dot(my_x)

        # Basically corresponds to Phi'(Gx) / Gx
        diag_term = 1.0 / (delta * np.sqrt(gx ** 2 + delta ** 2))
        mat_diag_term = diags(diag_term)
        gv = g.dot(my_v)
        right_term = g.T.dot(mat_diag_term.dot(gv))
        return hess_v + lamb * right_term


    # mm = MMQuadratic(f=f, qmajor=a_major_func, grad_f=grad_f, theta=1.9)
    # x_mm, stats_mm = mm.optimize(x0)
    # single_plot(stats_mm)
    # plt.show()

    ##############################
    ##############################
    # mg = ThreeMG(f=f, my_h=h, my_g=g, my_delta=delta, my_lambda=lamb,
    #              grad_f=grad_f)
    # x_mg, stats_mg = mg.optimize(x0=x0)
    # single_plot(stats_mg)
    # plt.show()


    ##############################
    ##############################
    # def get_aj(my_x, block_indices, coef=1.1):
    #     g_j = g[:, block_indices]  # Shape 2N, N_j
    #     my_x_j = x[block_indices].copy()
    #
    #     hess = h_transpose_h[:, block_indices]
    #     hess = hess[block_indices, :]
    #     assert hess.shape == (len(block_indices), len(block_indices))
    #     # I feel diag_psi_ddot  should contain
    #     # all indices if we go back
    #     # To the definition of a_j
    #     gx = g.dot(my_x)
    #     psi_ddot = delta * (delta ** 2 + gx ** 2) ** (-1.5)
    #     diag_psi_ddot = diags(psi_ddot).tocsc()
    #
    #     right_term = (coef * g_j).T.dot(diag_psi_ddot.dot(g_j))
    #     res = hess + right_term
    #     return res.todense()
    # from bloc_mm_quadratic import BlockMM
    #
    # b_mm = BlockMM(f=f, n_j=2700, grad_f=grad_f,
    #                get_aj=get_aj, n=N, max_iter=50)
    # x_bmm, stats_bmm = b_mm.optimize(x0)

    # NEW


    def get_aj(my_x, block_indices):
        x_j = my_x[block_indices].copy()
        g_j = g[:, block_indices]
        h_j = h[:, block_indices]
        g_x = g.dot(my_x)

        def a_j_func(vect):
            hess_v = h_j.T.dot(h_j.dot(vect))

            diag_term = 1.0 / (delta * np.sqrt(g_x ** 2 + delta ** 2))
            mat_diag_term = diags(diag_term)
            gv = g_j.dot(vect)
            right_term = g_j.T.dot(mat_diag_term.dot(gv))

            return hess_v + lamb * right_term

        return a_j_func

    from bloc_mm_quadratic import BlockMM

    b_mm = BlockMM(f=f, n_j=2700, grad_f=grad_f,
                   get_aj=get_aj, n=N, max_iter=50)
    x_bmm, stats_bmm = b_mm.optimize(x0)


def construct_f(my_lamb, my_delt):
    def my_f(my_x):
        first_vec = h.dot(my_x) - y
        first_vec = first_vec ** 2
        assert first_vec.shape == (16200,), first_vec.shape
        first_scal = 0.5 * first_vec.sum()

        g_x = g.dot(my_x)
        assert g_x.shape == (16200,)
        second_vec = np.sqrt(1.0 + (g_x ** 2) / (my_delt ** 2))
        second_scal = my_lamb * second_vec.sum()

        res = first_scal - second_scal
        return res

    return my_f


def construct_grad_f(my_lamb, my_delt):
    def my_grad_f(my_x):
        first_prod = h.dot(my_x) - y
        first_term = h.T.dot(first_prod)

        gx = g.dot(my_x)
        dot_phi = gx / (my_delt * np.sqrt(gx ** 2 + my_delt ** 2))

        grad = first_term + my_lamb * g.T.dot(dot_phi)
        return grad
    return my_grad_f


from sklearn.model_selection import ParameterGrid
params_grid = {
    'lambda': np.geomspace(1e-3, 1e0, 10),
    'delta': np.geomspace(1e-3, 1e-1, 10)
}
params = ParameterGrid(params_grid)
params = list(params)
params = [(param['lambda'], param['delta']) for param in params]


def parameters_search(x_init, my_params):
    scores = []
    for lamb_cand, delta_cand in my_params:
        new_f = construct_f(my_lamb=lamb_cand, my_delt=delta_cand)
        new_grad_f = construct_grad_f(my_lamb=lamb_cand, my_delt=delta_cand)

        mg = ThreeMG(f=new_f, my_h=h, my_g=g, my_delta=delta_cand, my_lambda=lamb_cand,
                     grad_f=new_grad_f, max_iter=500)
        x_mg, _ = mg.optimize(x0=x_init)
        scores.append(snr(x, x_mg))

    best_idx = np.argmax(scores)
    best_params = my_params[best_idx]
    best_score = scores[best_idx]

    return best_params, best_score
