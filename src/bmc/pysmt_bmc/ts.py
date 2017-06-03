""" Symbolic Transition system representation
"""

import logging
from pysmt.shortcuts import Symbol, Real
from pysmt.typing import BOOL, REAL
from pysmt.shortcuts import TRUE as TRUE_PYSMT
from pysmt.shortcuts import FALSE as FALSE_PYSMT
from pysmt.shortcuts import Not, And, Or, Implies, Iff, ExactlyOne, Equals
from pysmt.shortcuts import GE, LE
from pysmt.shortcuts import Plus, Times
from pysmt.shortcuts import get_env

from bmc.pysmt_bmc.helpers import Helper

class TransitionSystem:
    """ Symbolic transition system.
    All the objects are from PySMT (e.g. Symbols, formulas...)

    The TS is a tuple < state_var, init, trans >

    """
    def __init__(self, env=None, helper=None):
        if env is None:
            self.env = get_env()
            self.helper = Helper(self.env)
        else:
            self.env = env
            self.helper = helper
            assert (self.helper.env == env)

        # internal representation of the transition system
        self.state_vars = set()
        self.var_types = {}
        self.init = TRUE_PYSMT()
        self.trans = TRUE_PYSMT()

    def add_var(self, var):
        self.state_vars.add(var)

    def product(self, other_ts):
        """ Computes the synchronous product of self with other_ts,
        storing the product in self.

        Given TS1 = <V1, I1, T1> and TS2 = <V2, I2, T2>
        the product is the transition system
        TSP = <V1 union V2, I1 and I2, T1 and T2>

        (V are the state variables, I is the initial condition, T the transition relation)
        """

        self.state_vars.update(other_ts.state_vars)
        self.init = And(self.init, other_ts.init)
        self.trans = And(self.trans, other_ts.trans)

    def __repr__(self):
        """ Not efficient, need to use a buffer..."""

        res = "State vars: "
        for v in self.state_vars: res += ", %s" % v
        res += "\nINIT: "
        res += str(self.init.serialize())
        res += "\nTRANS: "
        res += str(self.trans.simplify().serialize())

        return res
