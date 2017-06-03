""" Translates a PWA in a symbolic TS """

from bmc.pysmt_bmc.ts import TransitionSystem
from bmc.pysmt_bmc.counter_enc import CounterEnc
from bmc.pysmt_bmc.helpers import Helper

import constraints
import pwa
from core.cellmodels import Qx
from core.modelrefine import ModelPartition

from pysmt.typing import BOOL, REAL
from pysmt.shortcuts import TRUE
from pysmt.shortcuts import FALSE
from pysmt.shortcuts import Not, And, Or, Implies, Iff, ExactlyOne, Equals
from pysmt.shortcuts import GE, LE
from pysmt.shortcuts import Plus, Times, Minus
from pysmt.shortcuts import Symbol, Real
from pysmt.shortcuts import get_env

from fractions import Fraction
from numpy import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, ndarray, isinf



class PWA2TS(object):
    def __init__(self, module_name, init_cons, final_cons, pwa_graph, vs, init_ps, final_ps):
        self.module_name = module_name
        assert isinstance(init_cons, constraints.IntervalCons)
        self.init_cons = init_cons
        assert isinstance(final_cons, constraints.IntervalCons)
        self.final_cons = final_cons
        assert isinstance(pwa_graph, pwa.pwagraph.PWAGraph)
        self.pwa_graph = pwa_graph
        assert isinstance(vs, list) and (not len(vs) > 0 or isinstance(vs[0],unicode))
        self.vs = vs
        assert isinstance(init_ps, set) and (not len(init_ps) > 0 or
                                             isinstance(next(iter(init_ps)),pwa.pwa.Partition))
        self.init_ps = init_ps
        assert isinstance(final_ps, set) and (not len(final_ps) > 0 or
                                              isinstance(next(iter(final_ps)),pwa.pwa.Partition))
        self.final_ps = final_ps

        self._ts = None

        # encoding of the graph locations
        self.pysmt_env = get_env()
        self.helper = Helper(self.pysmt_env)
        self._loc_enc = None
        self.val2loc = None # map from value to repr. node
        self.loc2val = None # map from loc to the encoding value
        self.pysmtvars = None

        self.pysmt2pwa_map = {}


    def _get_loc_var_name(self):
        return "loc"

    def _convert(self):
        self._ts = TransitionSystem(self.pysmt_env, self.helper)
        self._loc_enc = CounterEnc(self.pysmt_env)
        self.val2loc = {}
        self.loc2val = {}

        # Creates the boolean encoding for the location variables
        self._loc_enc.add_var(self._get_loc_var_name(),
                              len(self.pwa_graph.nodes()) - 1) # -1: it starts from 1
        for v in self._loc_enc.get_counter_var(self._get_loc_var_name()):
            self._ts.add_var(v)
            self._ts.var_types[v] = BOOL
        loc_id = -1
        for loc in self.pwa_graph.nodes():
            assert isinstance(loc, Qx)
            loc_id += 1
            self.val2loc[loc_id] = loc
            self.loc2val[loc] = loc_id

        # Add the continuous variables
        self.pysmtvars = []
        for var_name in self.vs:
            pysmt_var = Symbol(var_name, REAL)
            self.pysmtvars.append(pysmt_var)
            self._ts.add_var(pysmt_var)
            self._ts.var_types[pysmt_var] = REAL

        # Initial states
        self.init = self._convert_IntervalCons(self.init_cons)
        init_loc_smt = self._get_mod_part_set_enc(self.init_ps)
        self.init = And(self.init, init_loc_smt)

        # Final states
        final_smt = self._convert_IntervalCons(self.final_cons)
        final_loc_smt = self._get_mod_part_set_enc(self.final_ps)
        final_smt = And(final_smt, final_loc_smt)

        ## Location invariants
        loc_invars = TRUE()
        for loc in self.pwa_graph.nodes():
            assert loc.dim == len(self.vs) and loc.xdim == len(self.vs)
            assert isinstance(loc.ival_constraints, constraints.IntervalCons)
            loc_invar = self._convert_IntervalCons(loc.ival_constraints)
            loc_enc = self._get_loc_enc(loc)
            loc_invar = Or(Not(loc_enc), loc_invar)
            loc_invars = And(loc_invars, loc_invar)

        # Transition relation
        affine_trans_rel_smt = FALSE()
        for edge in self.pwa_graph.all_edges():
            assert isinstance(edge, tuple) and len(edge) == 2
            src_loc_smt = self._get_loc_enc(edge[0])
            dst_loc_smt = self._ts.helper.get_next_formula(self._ts.state_vars,
                                                           self._get_loc_enc(edge[0]))
            src_part = self.pwa_graph.node_p(edge[0])
            assert isinstance(src_part, ModelPartition)
            dst_part = self.pwa_graph.node_p(edge[0])
            assert isinstance(dst_part, ModelPartition)
            edge_rel = self.pwa_graph.edge_m(edge)
            assert isinstance(dst_part, ModelPartition)

            rel_enc = self._get_relation_enc(edge_rel)
            current_part = self._convert_partition(src_part)
            next_part = self._convert_partition(dst_part)
            next_part = self._ts.helper.get_next_formula(self._ts.state_vars,
                                                         next_part)

            affine_trans_rel_smt = Or(affine_trans_rel_smt,
                                      And(src_loc_smt, dst_loc_smt,
                                          current_part, next_part,
                                          rel_enc))

        assert (FALSE() != affine_trans_rel_smt)

        ## Transition system
        self.init = And(init_cons_smt, init_ps_smt)
        self.trans = And(loc_invars,
                         affine_trans_rel_smt)

    def _get_relation_enc(self, edge_rel):
        """ Encode x' = Ax + b +- error
        as 
        Ax + b - error <= x' /\ x' <= Ax + b + error
        """

        rel_enc = TRUE()
        for (var, i) in zip(self.pysmtvars,
                            range(len(edge_rel.A))):
            a_row = edge_rel.A[i]
            b_e_smt = self.to_real(edge_rel.b[i])
            e_e_l_smt = self.to_real(edge_rel.error.l[i])
            e_e_u_smt = self.to_real(edge_rel.error.h[i])
            var_next = Helper.get_next_var(var,
                                           self.pysmt_env.formula_manager)
            row_column = None
            for (var, a_el) in zip(self.pysmtvars, a_row):
                row_elem = Times(self.to_real(a_el), var)
                if row_column is None:
                    row_column = row_elem
                else:
                    row_column = Plus(row_column, row_elem)
            assert row_column is not None
            le = LE(var_next, Plus(row_column, b_e_smt, e_e_u_smt))
            ge = LE(Plus(row_column, Minus(b_e_smt, e_e_l_smt)), var_next)
            rel_enc = And(rel_enc, le)
            rel_enc = And(rel_enc, ge)

        return rel_enc

    def _get_mod_part_set_enc(self, loc_set):
        enc_loc_set_smt = FALSE()
        for model_partition in loc_set:
            mod_part_enc = self._get_mod_part_enc(model_partition)
            enc_loc_set_smt = Or(enc_loc_set_smt, mod_part_enc)
        return enc_loc_set_smt

    def _get_mod_part_enc(self, mod_part):
        assert isinstance(mod_part, ModelPartition)
        loc = mod_part.ID
        assert isinstance(loc, Qx)
        loc_enc = self._get_loc_enc(loc)
        # DEBUG
        # part_enc = self._convert_partition(mod_part)
        # mod_part_enc = And(part_enc, loc_enc)
        mod_part_enc = loc_enc
        return mod_part_enc        

    def _get_loc_enc(self, loc):
        loc_id = self.loc2val[loc]
        loc_enc = self._loc_enc.eq_val(self._get_loc_var_name(), loc_id)
        return loc_enc

    def _convert_IntervalCons(self, interval_cons):
        assert len(interval_cons.l) == len(self.pysmtvars)
        assert len(interval_cons.h) == len(self.pysmtvars)
        constraint = TRUE()
        for (var, l, h) in zip(self.pysmtvars, interval_cons.l, interval_cons.h):
            l_val = self.to_real(l)
            h_val = self.to_real(h)
            c = And(LE(l_val, var), LE(var, h_val))
            constraint = And(constraint, c)
        return constraint        

    def _convert_partition(self, partition):
        """ C x <= d """
        assert isinstance(partition, ModelPartition)
        assert len(self.pysmtvars) > 0

        print partition.C.shape

        assert len(self.pysmtvars) == len(partition.C)
        assert len(partition.C) == len(partition.d)
        enc_list = []
        for (x,c,d) in zip(self.pysmtvars, partition.C, partition.d):
            c_smt = self.to_real(c)
            d_smt = self.to_real(d)
            pred = LE(Plus(Times(x, c_smt)), d_smt)
            enc_list.append(pred)
        enc = Plus(enc_list)
        return enc
        

    def get_ts(self):
        """ Return the transition system """

        if self._ts is None:
            self._convert()
        assert self._ts is not None
        return self._ts

    
    def to_real(self, n):
        # TODO: use limit_denominator to approx the fraction
        # [SM] Warning: we need to consider the precision issue here.
        def is_numpy_int(n):
            return type(n) in [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
        def is_numpy_float(n):
            return type(n) in [float16, float32, float64]

        if isinf(n):
            raise ErrorInfinity
        elif is_numpy_float(n):
            frac_val = Fraction.from_float(n)
            return Real(frac_val)
        elif is_numpy_int(n):
            frac_val = Fraction(int(n),1)
            return Real(frac_val)
        else:
            # Let pysmt do the magic        
            return Real(frac_val)

