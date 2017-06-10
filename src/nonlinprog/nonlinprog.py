from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def factory(solver_str):
    if solver_str == 'z3':
        from nonlinprog.z3opt import nlinprog

    elif solver_str == 'scipy':
        from nonlinprog.scipyopt import nlinprog

    elif solver_str == 'ipopt':
        from nonlinprog.ipopt import nlinprog

    elif solver_str == 'pyfmincon':
        from nonlinprog.pyfmincon import nlinprog

    else:
        raise NotImplementedError('solver selected: {}'.format(solver_str))

    return nlinprog
