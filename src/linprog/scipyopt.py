from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.optimize as spopt

import linprog.spec as spec
import settings


def linprog(obj, A_ub, b_ub):
    """linprog

    Parameters
    ----------
    c : obj as a list
    A_ub : as an numpy array
    b_ub : as a numpy array

    Returns
    -------
    s

    Notes
    ------
    """
    # columns of A_ub
    num_opt_vars = A_ub.shape[1]
    bounds = [(-np.inf, np.inf)] * num_opt_vars

    disp_opt = True if settings.debug else False

    res = spopt.linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        method='simplex', options={'disp': disp_opt})

    return spec.OPTRES(res.fun, res.x, res.status, res.success)
