def factory(solver_str):
    if solver_str == 'z3':
        from nonlinprog.z3opt import nlinprog

    elif solver_str == 'dreal':
        from nonlinprog.dreal import nlinprog

    elif solver_str == 'scipy':
        from nonlinprog.scipyopt import nlinprog

    elif solver_str == 'ipopt':
        from nonlinprog.ipopt import nlinprog

    elif solver_str == 'pyfmincon':
        from nonlinprog.pyfmincon import nlinprog

    else:
        raise NotImplementedError(f'solver selected: {solver_str}')

    return nlinprog
