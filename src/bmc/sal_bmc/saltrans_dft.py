from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
Creates a SAL transition system: DFT
Each transition of the pwa system is encoded as a transition of SAL
'''

import textwrap as tw
from math import isinf
import itertools

from bmc.helpers.expr2str import Expr2Str
from globalopts import opts as gopts

import settings

##############################################
# ############ SAL Keywords ##################
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
##############################################


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


        #TODO:
        # Store a mapping from a cell id: tuple -> sal loc name: str
        self.partid2Cid = {}
        self.id_ctr = itertools.count()

        return


        #TODO:
    def get_C(self, partid):
        """Gets the bmc Cid corresponding to a pwa partition id
        Assumes all Cid have been added using add_C.
        Will raise an error if the requested location has not been
        added."""
        #return self.partid2Cid.setdefault(loc, 'C' + str(next(self.id_ctr)))
        return self.partid2Cid[partid]

        #TODO:
    def add_C(self, c):
        """Add a cell

        Parameters
        ----------
        c : cell id

        Notes
        ------
        Assumes every C is hashable and unique, else an
        overwrite will occur!
        """
        # TODO: Fix the ugliness using setdefault
        # simplified equivalent code from above
        if c not in self.partid2Cid:
            self.partid2Cid[c] = 'C' + str(next(self.id_ctr))
        return self.partid2Cid[c]

        #TODO:
    def add_Cs(self, Cs):
        """add_Cs
        Add all locations/cells in one-shot

        Parameters
        ----------
        Cs : cells

        Notes
        ------
        Assumes every location in locations is unique
        """
        # just to make sure
        Cs = set(Cs)
        self.partid2Cid = {c: 'C' + str(next(self.id_ctr)) for c in Cs}
        return



    def add_transition(self, tran):
        self.transitions.append(tran)

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
        return ';'.join(s)

    def init_set_def(self):
        return INIT + self.init_set

    # sal description
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
        return PROP.format(self.prop_name)

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
