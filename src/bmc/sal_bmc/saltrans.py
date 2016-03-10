'''
Creates a SAL transition system
'''

#import sympy as sm
import textwrap as tw

from math import isinf

# np.set_printoptions(suppress=True, precision=2)
# sp.set_printoptions(suppress=True, precision=2)
PREC = 2

class SALTransError(Exception):
    pass


# Make classes out of every header, prop, init, etc
class SALTransSys(object):

    def __init__(self, module_name, num_dim, init_set, prop):
        v = 'x'
        prop_name = 'safety'
        self.transitions = []
        self.prop = None
        self.trans = ''
        self.hdr = self.__class__.header_(module_name)
        self.init_set = self.__class__.init_set_(init_set, v)
        # TODO: ask for inputs and outputs
        self.decls = self.__class__.decls_(num_dim, v)
        self.prop = self.__class__.safety_prop_(prop_name, prop, v)
        return

    def add_init(self, init_set):
        '''adds the initial set.
           The set must be a hyper rectangular vector'''
        raise NotImplementedError

    def add_transition(self, tran):
        self.transitions.append(tran)

    def trans2str(self):
        ts = '[]\n'.join(('{}'.format(i) for i in self.transitions))
        self.trans = self.__class__.trans_(ts)

    def __str__(self):
        return self.sal_file()

    def sal_file(self):
        self.trans2str()
        s = tw.dedent('''
        {}
        OUTPUT{}
        INITIALIZATION{}
        TRANSITION
        [
        {}
        []
        TRUE -->
        x0' = x0;
        x1' = x1
        ]
        END;
        system: MODULE = PLANT;
        {}
        ''').format(self.hdr, self.decls, self.init_set, self.trans, self.prop)
        return s

    @staticmethod
    def header_(module_name):
        s = tw.dedent('''
        {}: CONTEXT =
        BEGIN

        PLANT: MODULE =
        BEGIN
        ''').format(module_name)
        return s

    @staticmethod
    def decls_(num_dim, v):
        s = ['\n\t{v}{i}:REAL' .format(v=v, i=i) for i in range(num_dim)]
        return ','.join(s)

    @staticmethod
    def init_set_(init_set, v):
        iv = init_set
        s = ['\n\t{v}{i} IN {{ r : REAL | r >=  {l} AND r <= {h} }}'.format(
            v=v, i=i, l=iv.l[i], h=iv.h[i]) for i in range(iv.dim)]
        return ';'.join(s)

    @staticmethod
    # sal description
    def trans_(transitions):
        s = '{}'.format(transitions)
        return s

    @staticmethod
    def safety_prop_(prop_name, prop, v):
        '''adds the property: is prop reachable?
           The prop must be a ival cons'''

        expr = '{v}{i} {gle} {c}'

        ls = [
                expr.format(v=v, i=i, gle='>=', c=prop.l[i]) for i in range(prop.dim)
                if not isinf(prop.l[i])
             ]
        hs = [
                expr.format(v=v, i=i, gle='<=', c=prop.h[i]) for i in range(prop.dim)
                if not isinf(prop.h[i])
             ]
        prop_str = ' AND '.join(ls + hs)

        s = tw.dedent('''
        {prop_name} : THEOREM
        system |- NOT F({prop});
        END
        ''').format(prop_name=prop_name, prop=prop_str)
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
        '''Cx <= d'''
        self.C = C
        self.d = d

    def __str__(self):
        # num_constraints , num_dimensions
        nc, nd = self.C.shape
        cons = ('{Cxi} <= {di}'.format(
            Cxi=' + '.join('{}*x{}'.format(self.C[i, j], j) for j in range(nd)),
            di=self.d[i])
                for i in range(nc))
        return ' AND '.join(cons)
        #return ' AND '.join('{}*x{} + {} <= 0'.format(self.C[i, j], j, -self.d[j]) for i in range(nc) for j in range(nd))

#     def __str__(self):
#         s = []
#         # num_constraints , num_dimensions
#         nc, nd = self.C.shape
#         x_sym = ['x{}'.format(i) for i in range(nd)]
#         x = sm.Matrix(sm.symbols(x_sym))
#         Cx = self.C * x
#         s = [str(Cx[i] - self.d[i] <= 0) for i in range(nc)]
#         return ' AND '.join(s)


class Reset(object):
    def __init__(self, A, b, tol=0):
        self.A = A
        self.b = b

        # use tolerance to change == into (<= and >=)
        if tol != 0:
            raise NotImplementedError

#     def __str__(self):
#         s = []
#         ndo, ndi = self.A.shape
#         x_sym = ['x{}'.format(i) for i in range(ndi)]
#         x__sym = ["x{}'".format(i) for i in range(ndo)]
#         x = sm.Matrix(sm.symbols(x_sym))
#         x_ = sm.Matrix(sm.symbols(x__sym))
#         Ax = self.A * x
#         s = ['{} = {}'.format(x_[i], Ax[i] + self.b[i]) for i in range(ndo)]
#         return ';\n'.join(s)

    def __str__(self):
        s = []
        ndo, ndi = self.A.shape

        xi_s = "x{}'"
        Axi_s = '{c:.{p}f}*x{j}'
        bi_s = '{c:.{p}f}'

        s = ['{xi_} = {Axi} + {bi}'.format(
            xi_=xi_s.format(i),
            Axi=' + '.join(Axi_s.format(p=PREC, c=self.A[i, j], j=j) for j in range(ndi)),
            bi=bi_s.format(p=PREC, c=self.b[i])) for i in range(ndo)]
        return ';\n'.join(s)

#         for i in range(ndo):
#             row = ['{}*x{}'.format(self.A[i, j], j) for j in range(ndi)]
#             # x_ = Ax + b
#             x__str = "x{}'".format(i)
#             Ax_str = ' + '.join(row)
#             b_str = '{}'.format(self.b[i])
#             eqn_str = '{} = {} + {}'.format(x__str, Ax_str, b_str)
#             s.append(eqn_str)
#         return ';\n'.join(s)
