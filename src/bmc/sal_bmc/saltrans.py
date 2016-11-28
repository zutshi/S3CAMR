from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
Creates a SAL transition system: DFT
'''
#TODO: rename to saltrans_dft.py

#import sympy as sm
import textwrap as tw
from math import isinf

from bmc.helpers.expr2str import Expr2Str
from globalopts import opts as gopts

#import utils as U
import settings


class SALTransError(Exception):
    pass


# Make classes out of every header, prop, init, etc
class SALTransSys(object):

    def __init__(self, module_name, vs, init_cons, prop):
        """
        Parameters
        ----------
        module_name : Module name string
        vs : list of variable names
        init_cons : X0 described as an interval constraint
        prop : Unsafe Set described as an interval constraint
        """

        self.vs = vs
        self.prop_name = 'safety'
        self.init_cons = init_cons
        self.module_name = module_name
        self.transitions = []
        self.prop = prop
        # initialize the class with the prec
        Expr2Str.set_prec(gopts.bmc_prec)
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
        % Always on transition: to overcome the in-complete pwa model
        % from deadlocking
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
        s = ['\n\t{v}:REAL' .format(v=v) for v in self.vs]
        return ','.join(s)

    @property
    def local_decls(self):
        return ''

    @property
    def init_set(self):
        iv = self.init_cons
        s = ['\n\t{v} IN {{ r : REAL | r >=  {l} AND r <= {h} }}'.format(
            v=v, l=Expr2Str.float2str(iv.l[i]), h=Expr2Str.float2str(iv.h[i])) for i, v in enumerate(self.vs)]
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
        expr = "{v}' {gle} {c}"

        ls = [
                expr.format(v=v, gle='>=', c=prop.l[i])
                for i, v in enumerate(self.vs)
                if not isinf(prop.l[i])
             ]
        hs = [
                expr.format(v=v, gle='<=', c=prop.h[i])
                for i, v in enumerate(self.vs)
                if not isinf(prop.h[i])
             ]
        prop_str = ' AND '.join(ls + hs)

        # Remove AND cell' = CE
        assert(settings.CE)
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
        unsafe' IN {{r : BOOLEAN | r <=> ({} AND cell' = CE)}}
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
        '''Cx - d <= 0'''
        self.C = C
        self.d = d

        self.ncons, self.nvars = self.C.shape
        assert(self.ncons == d.size)
        self._vs = None

    @property
    def vs(self):
        if self._vs is None:
            return ['x'+str(j) for j in range(self.nvars)]
        else:
            return self._vs

    @vs.setter
    def vs(self, vs):
        self._vs = vs

    def __str__(self):
        # num_constraints , num_dimensions
        #nc, nd = self.C.shape
        #TODO: Remove this
        # num states
        #num_x = nr
        # num ex. inputs
        #num_w = nc - nr
        cons = (Expr2Str.linexpr2str(self.vs, self.C[i, :], -self.d[i]) + ' <= 0'
                for i in range(self.ncons))
        return ' AND '.join(cons)


class Reset(object):
    def __init__(self, A, b, error):
        self.A = A
        self.b = b
        self.e = error

        self.nlhs, self.nrhs = self.A.shape
        assert(self.nlhs == b.size)
        self._vs = None
        self._vs_ = None

        #TODO: Remove this
        # num op, num ip
        #nr, nc = self.A.shape
        # num states
        #self.num_x = nr
        # num ex. inputs
        #self.num_w = nc - nr

    @property
    def vs(self):
        if self._vs is None:
            return ['x'+str(j) for j in range(self.nrhs)]
        else:
            return self._vs

    @property
    def vs_(self):
        if self._vs_ is None:
            return ["x{}".format(i) for i in range(self.nlhs)]
        else:
            return self._vs_

    @vs.setter
    def vs(self, vs):
        self._vs = vs

    @vs_.setter
    def vs_(self, vs_):
        self._vs_ = vs_

    def __str__(self):
        s = []

        delta_h = self.b + self.e.h
        delta_l = self.b + self.e.l
        set_def = ("{xi_}' IN {{ r : REAL| "
                   "r >= {Axi_plus_delta_li} AND "
                   "r <= {Axi_plus_delta_hi} }}")
        for vsi_, Ai, dli, dhi in zip(self.vs_, self.A, delta_l, delta_h):
            #if error is not significant, i.e., it is less than the
            # precision used while dumping sal file, ignore it. Else,
            # include it and make the rhs of the assignment a set.
            if Expr2Str.float2str(dli) == Expr2Str.float2str(dhi):
                assignment_stmt = vsi_ + "' = " + Expr2Str.linexpr2str(self.vs, Ai, dli)
            else:
                assignment_stmt = set_def.format(
                        xi_=vsi_,
                        Axi_plus_delta_li=Expr2Str.linexpr2str(self.vs, Ai, dli),
                        Axi_plus_delta_hi=Expr2Str.linexpr2str(self.vs, Ai, dhi)
                        )
            s.append(assignment_stmt)
        return ';\n'.join(s)


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

# Older code, [same as new one but with comments]
# class Guard(object):
#     def __init__(self, C, d):
#         '''Cx <= d'''
#         self.C = C
#         self.d = d

#     def __str__(self):
#         # num_constraints , num_dimensions
#         nc, nd = self.C.shape

#         xs = ['x'+str(j) for j in range(nd)]
#         cons = (linexpr2str(xs, self.C[i, :], -self.d[i]) + '<= 0'
#                 for i in range(nc))
# #         cons = ('{Cxi} <= {di}'.format(
# #             Cxi=linexpr2str(self.C[i, :], ('x'+str(j) for j in range(nd))),
# #             di=float2str(self.d[i])) for i in range(nc))
#         return ' AND '.join(cons)
#         #return ' AND '.join('{}*x{} + {} <= 0'.format(self.C[i, j], j, -self.d[j]) for i in range(nc) for j in range(nd))

# #     def __str__(self):
# #         s = []
# #         # num_constraints , num_dimensions
# #         nc, nd = self.C.shape
# #         x_sym = ['x{}'.format(i) for i in range(nd)]
# #         x = sm.Matrix(sm.symbols(x_sym))
# #         Cx = self.C * x
# #         s = [str(Cx[i] - self.d[i] <= 0) for i in range(nc)]
# #         return ' AND '.join(s)


# class Reset(object):
#     def __init__(self, A, b, error=None):
#         self.A = A
#         self.b = b
#         self.e = error
# #     def __str__(self):
# #         s = []
# #         ndo, ndi = self.A.shape
# #         x_sym = ['x{}'.format(i) for i in range(ndi)]
# #         x__sym = ["x{}'".format(i) for i in range(ndo)]
# #         x = sm.Matrix(sm.symbols(x_sym))
# #         x_ = sm.Matrix(sm.symbols(x__sym))
# #         Ax = self.A * x
# #         s = ['{} = {}'.format(x_[i], Ax[i] + self.b[i]) for i in range(ndo)]
# #         return ';\n'.join(s)

#     def __str__(self):
#         s = []
#         ndo, ndi = self.A.shape

#         #xi_s = "x{}'"
#         xi_s = ["x{}'".format(i) for i in range(ndo)]
#         xs = ['x'+str(j) for j in range(ndi)]

#         if self.e is None:
#             s = (xi_s[i] + '=' + linexpr2str(xs, self.A[i, :], self.b[i])
#                  for i in range(ndo))
#             return ';\n'.join(s)
# #             s = ['{xi_} = {Axi} + {bi}'.format(
# #                 xi_=xi_s.format(i),
# #                 Axi=linexpr2str(
# #                     self.A[i, :], ('x'+str(j) for j in range(ndi))
# #                     ),
# #                 bi=float2str(self.b[i])
# #                 ) for i in range(ndo)
# #                 ]

#         else:
#             delta_h = self.b + self.e.h
#             delta_l = self.b + self.e.l

#             s = ('{xi_} IN {{ r : REAL| '
#                  'r >= {Axi_plus_delta_li} AND '
#                  'r <= {Axi_plus_delta_hi} }}'
#                  .format(
#                      xi_=xi_s[i],
#                      Axi_plus_delta_li=linexpr2str(xs, self.A[i, :], delta_l[i]),
#                      Axi_plus_delta_hi=linexpr2str(xs, self.A[i, :], delta_h[i])
#                      ) for i in range(ndo)
#                  )

# #             s = ['{xi_} IN {{ r : REAL| '
# #                  'r >= {Axi} + {delta_li} AND '
# #                  'r <= {Axi} + {delta_hi} }}'
# #                  .format(
# #                     xi_=xi_s.format(i),
# #                     Axi=linexpr2str(
# #                         self.A[i, :], ('x'+str(j) for j in range(ndi))
# #                         ),
# #                     delta_li=float2str(delta_l[i]),
# #                     delta_hi=float2str(delta_h[i])
# #                     ) for i in range(ndo)
# #                  ]
#             return ';\n'.join(s)


# #         for i in range(ndo):
# #             row = ['{}*x{}'.format(self.A[i, j], j) for j in range(ndi)]
# #             # x_ = Ax + b
# #             x__str = "x{}'".format(i)
# #             Ax_str = ' + '.join(row)
# #             b_str = '{}'.format(self.b[i])
# #             eqn_str = '{} = {} + {}'.format(x__str, Ax_str, b_str)
# #             s.append(eqn_str)
# #         return ';\n'.join(s)
