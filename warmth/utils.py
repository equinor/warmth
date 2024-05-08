# from numba import jit, prange
# import gzip
from pathlib import Path
import pickle
import numpy as np
from scipy import interpolate

from .logging import logger
# https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4


# # @jit(nogil=True, fastmath=True)
# def _coeff_mat(x, deg):
#     mat_ = np.zeros(shape=(x.shape[0], deg + 1))
#     const = np.ones_like(x)
#     mat_[:, 0] = const
#     mat_[:, 1] = x
#     if deg > 1:
#         for n in range(2, deg + 1):
#             mat_[:, n] = x**n
#     return mat_


# # @jit(nogil=True, fastmath=True)
# def _fit_x(a, b):
#     # linalg solves ax = b
#     det_ = np.linalg.lstsq(a, b)[0]
#     return det_


# # @jit(nogil=True, fastmath=True)
# def fit_poly(x, y, deg):
#     a = _coeff_mat(x, deg)
#     p = _fit_x(a, y)
#     # Reverse order so p[0] is coefficient of highest order
#     return p[::-1]


# # @jit(nogil=True, fastmath=True)
# def eval_polynomial(P, x):
#     '''
#     Compute polynomial P(x) where P is a vector of coefficients, highest
#     order coefficient at P[0].  Uses Horner's Method.
#     '''
#     result = 0
#     for coeff in P:
#         result = x * result + coeff
#     return result


# # @jit(nogil=True, fastmath=True, parallel=True)
# def eval_polynomial_arr(P, arr_x):
#     size = arr_x.size
#     result_arr_y = np.empty(size)
#     for i in range(size):
#         result = 0
#         for coeff in P:
#             result = arr_x[i] * result + coeff
#         result_arr_y[i] = result
#     return result_arr_y


# def reshape_ndarray(arr, new_shape):
#     shape_arr_nd = sum(new_shape[1:])
#     if arr.size == shape_arr_nd:
#         arr = np.repeat(arr, new_shape[0], axis=0)
#         arr = arr.reshape(new_shape)
#     elif arr.size == new_shape[0]:
#         arr = np.repeat(arr, shape_arr_nd)
#         arr = arr.reshape(new_shape)
#     return arr
# #@jit
# def brents(f, x0, x1, args=(), max_iter=50, tolerance=1e-5):

#     fx0 = f(x0)
#     fx1 = f(x1)

#     assert (fx0 * fx1) <= 0, "Root not bracketed"

#     if abs(fx0) < abs(fx1):
#         x0, x1 = x1, x0
#         fx0, fx1 = fx1, fx0

#     x2, fx2 = x0, fx0

#     mflag = True
#     steps_taken = 0

#     while steps_taken < max_iter and abs(x1 - x0) > tolerance:
#         fx0 = f(x0)
#         fx1 = f(x1)
#         fx2 = f(x2)

#         if fx0 != fx2 and fx1 != fx2:
#             L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
#             L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
#             L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
#             new = L0 + L1 + L2

#         else:
#             new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

#         if (
#             (new < ((3 * x0 + x1) / 4) or new > x1)
#             or (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2))
#             or (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2))
#             or (mflag == True and (abs(x1 - x2)) < tolerance)
#             or (mflag == False and (abs(x2 - d)) < tolerance)
#         ):
#             new = (x0 + x1) / 2
#             mflag = True

#         else:
#             mflag = False

#         fnew = f(new)
#         d, x2 = x2, x1

#         if (fx0 * fnew) < 0:
#             x1 = new
#         else:
#             x0 = new

#         if abs(fx0) < abs(fx1):
#             x0, x1 = x1, x0

#         steps_taken += 1

#     return x1, steps_taken

def compressed_pickle_save(data, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def compressed_pickle_open(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_pickle(filepath: Path):
    logger.info(f"Loading data from {filepath}")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def interpolator(x, y, val, grid):
    grid_x, grid_y = np.mgrid[
        grid.origin_x: grid.xmax+grid.step_x: grid.step_x,
        grid.origin_yn: grid.ymax+grid.step_y: grid.step_y,
    ]
    rbfi = interpolate.Rbf(x, y, val)
    di = rbfi(grid_x, grid_y)
    return di
