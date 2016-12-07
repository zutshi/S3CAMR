from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import glpk

# Tuple contains lp results. It follows the scipy convention
OPTRES = collections.namedtuple('optres', ('fun', 'status', 'success'))


# TODO: make it handle multiple objectives?
# Specifially if just changing an obj offers a warm restart? It will
# also eschew the time required to re-create the LP. WIll need to
# research if such an optimization is legal and does it actually save
# enough time?
def linprog(obj, A, b):
    """linprog

    Parameters
    ----------
    c : obj as a list
    A : as an numpy array
    b : as a numpy array

    Returns
    -------
    s

    Notes
    ------
    """
    lp = glpk.LPX()
    lp.name = 'min x0'
    lp.obj.maximize = False

    lp.rows.add(len(b))

    for idx, r in enumerate(lp.rows):
        r.name = 'r{}'.format(idx)
        r.bounds = None, b[idx]

    lp.cols.add(A.shape[1])

    for idx, c in enumerate(lp.cols):
        c.name = 'x{}'.format(idx)
        c.bounds = None, None

    lp.obj[:] = obj.tolist()

    lp.matrix = A.flatten()
    lp.simplex()
    #lp.write(prob='glpk_out')
    return OPTRES(lp.obj.value, lp.status, lp.status == 'opt')
