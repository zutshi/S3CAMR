'''
Creates a SAL transition system: DFT
'''
#TODO: rename to saltrans_dft.py

#import sympy as sm
import textwrap as tw
from math import isinf

from ..helpers.expr2str import linexpr2str


class SALTransError(Exception):
    pass


# Make classes out of every header, prop, init, etc
class SALTransSys(object):

    def __init__(self, module_name, num_dim, init_cons, prop):
        '''adds the initial set.
           The set must be an ival cons (hyper rectangular vector)

           adds the property: is prop reachable?
           The prop must be an ival cons'''

        self.num_dim = num_dim
        self.v = 'x'
        self.prop_name = 'safety'
        self.init_cons = init_cons
        self.module_name = module_name
        self.transitions = []
        self.prop = prop
        # TODO: ask for inputs and outputs
        return

    def add_transition(self, tran):
        self.transitions.append(tran)

    def __str__(self):
        return self.sal_file

    # XXX
    # This is a bit strange, Some of the declarations like OUTPUT and
    # INITIALIZATION are present but others like LOCAL are absent. The
    # one whcich are mandatory (in the present context) are present
    # and the ones which are optional like LOCAL are specified by the
    # variables. This can be fixed later by making this a class.
    @property
    def sal_file(self):
        s = tw.dedent('''
        {c}
        {tp}
        {pm}
        {ld}
        OUTPUT
        {od}
        INITIALIZATION
        {init}
        TRANSITION
        [
        {trans}
        {atran}
        ]
        END;
        {monitor}
        system: MODULE = PLANT || MONITOR;
        {prop}
        END
        ''').format(c=self.context,
                    tp=self.type_decls,
                    pm=self.plant_module,
                    ld=self.local_decls,
                    od=self.op_decls,
                    init=self.init_set,
                    trans=self.trans,
                    atran=self.always_true_transition,
                    monitor=self.monitor_module,
                    prop=self.safety_prop)
        return s

    @property
    def always_true_transition(self):
        return '\n[] NOP: TRUE -->\n'

    @property
    def context(self):
        s = tw.dedent('''
        {}: CONTEXT =
        BEGIN
        ''').format(self.module_name)
        return s

    @property
    def type_decls(self):
        # There are no type decls
        return ''

    @property
    def plant_module(self):
        s = tw.dedent('''
        PLANT: MODULE =
        BEGIN''')
        return s

    @property
    def op_decls(self):
        nd = self.num_dim
        v = self.v
        s = ['\n\t{v}{i}:REAL' .format(v=v, i=i) for i in range(nd)]
        return ','.join(s)

    @property
    def local_decls(self):
        return ''

    @property
    def init_set(self):
        iv = self.init_cons
        v = self.v
        s = ['\n\t{v}{i} IN {{ r : REAL | r >=  {l} AND r <= {h} }}'.format(
            v=v, i=i, l=iv.l[i], h=iv.h[i]) for i in range(iv.dim)]
        return ';'.join(s)

    # sal description
    @property
    def trans(self):
        ts = '[]\n'.join(('{}'.format(i) for i in self.transitions))
        s = '{}'.format(ts)
        return s

    @property
    def safety_prop(self):
        s = tw.dedent('''
        {prop_name} : THEOREM
        system |- G(NOT unsafe);
        ''').format(prop_name=self.prop_name)
        return s

    @property
    def monitor_module(self):
        prop = self.prop
        v = self.v
        expr = "{v}{i}' {gle} {c}"

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
        MONITOR: MODULE =
        BEGIN
        OUTPUT
                unsafe : BOOLEAN
        INPUT
        {}
        INITIALIZATION
                unsafe = FALSE
        TRANSITION
        [
        TRUE -->
        unsafe' IN {{r : BOOLEAN | r <=> ({})}}
        ]
        END;''').format(self.op_decls, prop_str)
        return s


class Transition(object):

    def __init__(self, name, g, r):
        self.g = g
        self.r = r
        self.name = name

    def __str__(self):
        s = '{}:\n{} -->\n{}\n'.format(self.name, self.g, self.r)
        return s


class Guard(object):
    def __init__(self, C, d):
        '''Cx <= d'''
        self.C = C
        self.d = d

    def __str__(self):
        # num_constraints , num_dimensions
        nc, nd = self.C.shape

        xs = ['x'+str(j) for j in range(nd)]
        cons = (linexpr2str(xs, self.C[i, :], -self.d[i]) + '<= 0'
                for i in range(nc))
#         cons = ('{Cxi} <= {di}'.format(
#             Cxi=linexpr2str(self.C[i, :], ('x'+str(j) for j in range(nd))),
#             di=float2str(self.d[i])) for i in range(nc))
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
    def __init__(self, A, b, error=None):
        self.A = A
        self.b = b
        self.e = error
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

        #xi_s = "x{}'"
        xi_s = ["x{}'".format(i) for i in range(ndo)]
        xs = ['x'+str(j) for j in range(ndi)]

        if self.e is None:
            s = (xi_s[i] + '=' + linexpr2str(xs, self.A[i, :], self.b[i])
                 for i in range(ndo))
            return ';\n'.join(s)
#             s = ['{xi_} = {Axi} + {bi}'.format(
#                 xi_=xi_s.format(i),
#                 Axi=linexpr2str(
#                     self.A[i, :], ('x'+str(j) for j in range(ndi))
#                     ),
#                 bi=float2str(self.b[i])
#                 ) for i in range(ndo)
#                 ]

        else:
            delta_h = self.b + self.e.h
            delta_l = self.b + self.e.l

            s = ('{xi_} IN {{ r : REAL| '
                 'r >= {Axi_plus_delta_li} AND '
                 'r <= {Axi_plus_delta_hi} }}'
                 .format(
                     xi_=xi_s[i],
                     Axi_plus_delta_li=linexpr2str(xs, self.A[i, :], delta_l[i]),
                     Axi_plus_delta_hi=linexpr2str(xs, self.A[i, :], delta_h[i])
                     ) for i in range(ndo)
                 )

#             s = ['{xi_} IN {{ r : REAL| '
#                  'r >= {Axi} + {delta_li} AND '
#                  'r <= {Axi} + {delta_hi} }}'
#                  .format(
#                     xi_=xi_s.format(i),
#                     Axi=linexpr2str(
#                         self.A[i, :], ('x'+str(j) for j in range(ndi))
#                         ),
#                     delta_li=float2str(delta_l[i]),
#                     delta_hi=float2str(delta_h[i])
#                     ) for i in range(ndo)
#                  ]
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


###############################################
# ############# CEMETEREY
###############################################

#     @property
#     def safety_prop(self):
#         prop = self.prop
#         v = self.v
#         expr = '{v}{i} {gle} {c}'

#         ls = [
#                 expr.format(v=v, i=i, gle='>=', c=prop.l[i]) for i in range(prop.dim)
#                 if not isinf(prop.l[i])
#              ]
#         hs = [
#                 expr.format(v=v, i=i, gle='<=', c=prop.h[i]) for i in range(prop.dim)
#                 if not isinf(prop.h[i])
#              ]
#         prop_str = ' AND '.join(ls + hs)

#         s = tw.dedent('''
#         {prop_name} : THEOREM
#         system |- NOT F({prop});
#         END''').format(prop_name=self.prop_name, prop=prop_str)
#         return s
