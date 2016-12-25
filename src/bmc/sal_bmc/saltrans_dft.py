from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
Creates a SAL transition system: DFT
Each transition of the pwa system is encoded as a transition of SAL
'''

import itertools as it

import textwrap as tw

from bmc.helpers.expr2str import Expr2Str
from globalopts import opts as gopts

##############################################
# ############ SAL Skeleton ##################
HDR = tw.dedent('''
        {}: CONTEXT =
        BEGIN
        ''')
MODULE = tw.dedent('''
            {}: MODULE =
            BEGIN''')
OUTPUT = 'OUTPUT'
LOCAL = 'LOCAL'
INIT = 'INITIALIZATION'
TRAN = 'TRANSITION'
PROP = tw.dedent('''
        {} : THEOREM
        system |- G(NOT unsafe);
        ''')

PROP_NAME = 'safety'
##############################################


# Make classes out of every header, prop, init, etc
class SALTransSys(object):

    def __init__(self, module_name, vs,
                 init_cons, final_cons,
                 init_ps, final_ps,
                 transitions, partid2Cid):
        """
        Parameters
        ----------
        module_name : Module name string
        vs : list of variable names
        init_cons : X0 described as an interval constraint
        final_cons : final Set described as an interval constraint
        """

        self.vs = vs
        self.vs_ = [vi + "'" for vi in vs]
        self.module_name = module_name

        self.init_cons = init_cons
        self.final_cons = final_cons
        self.init_ps = init_ps
        self.final_ps = final_ps

        self.transitions = transitions
        # initialize the class with the prec
        Expr2Str.set_prec(gopts.bmc_prec)
        self.partid2Cid = partid2Cid

    def __str__(self):
        return self.sal_file

    @property
    def ncells(self):
        return len(self.partid2Cid)

    @property
    def sal_file(self):
        s = tw.dedent('''
        {c}
        {tp}
        {pm}
        {ld}
        {od}
        {init}
        {trans}
        END;
        {monitor}
        system: MODULE = PLANT || MONITOR;
        {prop}
        END
        ''').format(c=self.context(),
                    tp=self.type_def(),
                    pm=self.plant_module(),
                    ld=self.local_def(),
                    od=self.op_def(),
                    init=self.init_set_def(),
                    trans=self.trans_def(),
                    monitor=self.monitor_module(),
                    prop=self.safety_prop)
        return s

    def context(self):
        return HDR.format(self.module_name)

    @property
    def custom_types(self):
        s = ('\nCELL: TYPE = {' + ', '.join(self.partid2Cid.values()) + '};'
             if self.ncells > 0 else '')
        return s

    def type_def(self):
        return self.custom_types

    def plant_module(self):
        return MODULE.format('PLANT')

    @property
    def outputs(self):
        s = ['\n\t{v}:REAL' .format(v=v) for v in self.vs]
        cells = ',\n\tcell:CELL' if self.ncells else ''
        return ','.join(s) + cells

    def op_def(self):
        return (OUTPUT + self.outputs) if self.outputs else ''

    @property
    def locals_(self):
        return ''

    def local_def(self):
        return (LOCAL + self.locals_) if self.locals_ else ''

    @property
    def init_set(self):
        iv = self.init_cons
        s = ['\n\t{v} IN {{ r : REAL | r >=  {l} AND r <= {h} }}'.format(
            v=v, l=Expr2Str.float2str(iv.l[i]), h=Expr2Str.float2str(iv.h[i])) for i, v in enumerate(self.vs)]
        init_cons_str = ';'.join(s)

        init_cells = [self.partid2Cid[p.ID] for p in self.init_ps]
        init_cells_str = '\n\tcell IN {' + ', '.join(init_cells) + '}'

        return init_cons_str + ';' + init_cells_str

    def init_set_def(self):
        return INIT + self.init_set

    @property
    def trans(self):
        ts = '[]\n'.join(('{}'.format(i) for i in self.transitions))
        s = '{}'.format(ts)

        atran = tw.dedent('''
           % Always on transition: to overcome the in-complete pwa model
           % from deadlocking
           [] NOP: TRUE -->
           ''')
        return s + atran

    def trans_def(self):
        lbracket = '\n[\n'
        rbracket = ']\n'
        return TRAN + lbracket + self.trans + rbracket

    @property
    def safety_prop(self):
        return PROP.format(PROP_NAME)

    def monitor_module(self):
        state_cons = ' AND '.join(it.chain(
                                *self.final_cons.linexpr_str(self.vs_)))

        final_cells = (self.partid2Cid[p.ID] for p in self.final_ps)
        cell_cons = ' OR '.join("cell' = {}".format(c) for c in final_cells)

        prop_str = state_cons + ' AND ' + '({})'.format(cell_cons)

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
        END;''').format(self.outputs, prop_str)
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
    def __init__(self, conjuncts):
        self.conjuncts = conjuncts

    def __str__(self):
        return ' AND '.join(self.conjuncts)


class Reset(object):
    def __init__(self, assignments):
        self.assignments = assignments

    def __str__(self):
        return ';\n'.join(self.assignments)
