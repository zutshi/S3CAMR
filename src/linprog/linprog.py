from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


def factory(solver_str):
    if solver_str == 'glpk':
        import linprog.pyglpklp as pyglpklp
        #res = [pyglpklp.linprog(obj, A_ub, b_ub) for obj in directions_ext]
        #lp_fun = ft.partial(pyglpklp.linprog, A_ub=A_ub, b_ub=b_ub)
        lp_fun = pyglpklp.linprog

    elif solver_str == 'gurobi':
        import linprog.pygurobi as pygurobi
        #res = [pygurobi.linprog(obj, A_ub, b_ub) for obj in directions_ext]
        #lp_fun = ft.partial(pygurobi.linprog, A_ub=A_ub, b_ub=b_ub)
        lp_fun = pygurobi.linprog

    elif solver_str == 'scipy':
        import linprog.scipyopt as scipyopt
        #lp_fun = ft.partial(scipyopt.linprog, A_ub=A_ub, b_ub=b_ub)
        lp_fun = scipyopt.linprog

    else:
        raise NotImplementedError('solver selected: {}'.format(solver_str))

    return lp_fun
