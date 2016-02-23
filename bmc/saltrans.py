'''
Creates a SAL transition system
'''

import sympy as sm
import numpy as np

np.set_printoptions(suppress=True, precision=2)



class SALTransError(Exception):
    pass


# sal description
def sal_dsc(transitions, prop):
    s = '''
    vdp: CONTEXT =
    BEGIN

    PLANT: MODULE =
    BEGIN
    OUTPUT  x0:REAL, x1:REAL
    %OUTPUT y1:REAL, y2:REAL

    INITIALIZATION
            x0 IN {{ r : REAL | r >= -0.4 AND r <= 0.4 }} ;
            x1 IN {{ r : REAL | r >= -0.4 AND r <= 0.4 }} ;

    TRANSITION
    [
    {transitions}
    ]
    END;
    system: MODULE = PLANT;

    safety : THEOREM
    system |- {prop};
    END

    '''.format(transitions=transitions, prop=prop)
    return s


class SALTransSys(object):

    def __init__(self):
        self.transitions = []
        self.prop = None

    def add_prop(self, prop):
        self.prop = prop

    def add_transition(self, tran):
        self.transitions.append(tran)

    def __str__(self):
        ts = '[]\n'.join(('{}'.format(i) for i in self.transitions))
        s = sal_dsc(ts, self.prop)
        return s


class Transition(object):

    def __init__(self, g, r):
        self.g = g
        self.r = r

    def __str__(self):
        s = '{} -->\n{}\n'.format(self.g, self.r)
        return s


class Guard(object):
    def __init__(self, C, d):
        self.C = C
        self.d = d

    def __str__(self):
        s = []
        # num_constraints , num_dimensions
        nc, nd = self.C.shape
        x_sym = ['x{}'.format(i) for i in range(nd)]
        x = sm.Matrix(sm.symbols(x_sym))
        Cx = self.C * x
        s = [str(Cx[i] - self.d[i] <= 0) for i in range(nc)]
        return ' AND '.join(s)


class Reset(object):
    def __init__(self, A, b, tol=0):
        self.A = A
        self.b = b

        # use tolerance to change == into (<= and >=)
        if tol != 0:
            raise NotImplementedError

    def __str__(self):
        s = []
        ndo, ndi = self.A.shape
        x_sym = ['x{}'.format(i) for i in range(ndi)]
        x__sym = ["x{}'".format(i) for i in range(ndo)]
        x = sm.Matrix(sm.symbols(x_sym))
        x_ = sm.Matrix(sm.symbols(x__sym))
        Ax = self.A * x
        s = ['{} = {}'.format(x_[i], Ax[i] + self.b[i]) for i in range(ndo)]
        return ';\n'.join(s)
