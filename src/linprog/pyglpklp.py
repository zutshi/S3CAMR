from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import glpk

#from IPython import embed

# Tuple contains lp results. It follows the scipy convention
OPTRES = collections.namedtuple('optres', ('fun', 'x', 'status', 'success'))

EPS = 1e-5


# TODO: make it handle multiple objectives?
# Specifially if just changing an obj offers a warm restart? It will
# also eschew the time required to re-create the LP. WIll need to
# research if such an optimization is legal and does it actually save
# enough time?
def linprog(obj, A_ub, b_ub, exact=False):
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

    #b_ub += EPS

    lp = glpk.LPX()
    lp.name = 'min x0'
    lp.obj.maximize = False

    lp.rows.add(len(b_ub))

    for idx, r in enumerate(lp.rows):
        r.name = 'r{}'.format(idx)
        r.bounds = None, b_ub[idx]

    lp.cols.add(A_ub.shape[1])

    for idx, c in enumerate(lp.cols):
        c.name = 'x{}'.format(idx)
        c.bounds = None, None

    lp.obj[:] = obj.tolist()

    lp.matrix = A_ub.flatten()
    if exact:
        lp.exact()
    else:
        lp.simplex()

    # doenst work for some reason: solution is 0 all the time
    #kkt = lp.kkt()
    #print(kkt.pe_ae_max, kkt.pe_ae_row, kkt.pe_quality, kkt.pe_re_max, kkt.pe_re_row)
    #print(kkt.de_ae_max, kkt.de_ae_row, kkt.de_quality, kkt.de_re_max, kkt.de_re_row)
    #embed()

    #lp.write('glpk_out')
    #lp.write(prob='glpk_out')
    x = lp.cols
    return OPTRES(lp.obj.value, x, lp.status, lp.status == 'opt')
