""" Translates a PWA in a symbolic TS """

# from bmc.pysmt_bmc.ts import TransitionSystem
# from bmc.pysmt_bmc.counter_enc import CounterEnc
# from bmc.pysmt_bmc.helpers import Helper

from ts import TransitionSystem
from counter_enc import CounterEnc
from helpers import Helper


import constraints
import pwa
from core.cellmodels import Qx
from core.modelrefine import ModelPartition

import pysmt.operators as op
from pysmt.typing import BOOL, REAL
# Adi: Faster to obtain from env than using shortcuts
#from pysmt.shortcuts import TRUE
#from pysmt.shortcuts import FALSE
#from pysmt.shortcuts import Not, And, Or, Implies, Iff, ExactlyOne, Equals
#from pysmt.shortcuts import GE, LE
#from pysmt.shortcuts import Plus, Times, Minus
#from pysmt.shortcuts import Symbol, Real
from pysmt.shortcuts import get_env

from fractions import Fraction
from numpy import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, ndarray, isinf


# Adi: Faster than using shortcuts when using in a loop
Symbol = get_env().formula_manager.Symbol
Real = get_env().formula_manager.Real
GE = get_env().formula_manager.GE
LE = get_env().formula_manager.LE
Plus = get_env().formula_manager.Plus
Times = get_env().formula_manager.Times
Minus = get_env().formula_manager.Minus
Not = get_env().formula_manager.Not
And = get_env().formula_manager.And
Or = get_env().formula_manager.Or
Implies = get_env().formula_manager.Implies
Iff = get_env().formula_manager.Iff
ExactlyOne = get_env().formula_manager.ExactlyOne
Equals = get_env().formula_manager.Equals
TRUE = get_env().formula_manager.TRUE
FALSE = get_env().formula_manager.FALSE


class ErrorInfinity(Exception):
    pass


class PWA2TS(object):
    """ Translates a PWA in a symbolic TS

    Usage
    converter = PWA2TS(module_name, init_cons, final_cons, pwa_graph, vs, init_ps, final_ps)
    ts = self.converter.get_ts()

    """


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

        self._loc_enc = None # Object used to encode the location with a counter
        self.val2loc = None # map from value to locations (as represented in the graph)
        self.loc2val = None # map from location (as rep. in graph) to the encoding value
        self.val2edge = None # map from value to transition
        self.edge2val = None # map from transition to value
        self.pysmtvars = None # List of pysmt variables (same position as self.vs)
        self.pysmtvars2index = {} # map from variables in the ts to the index used in vs

        self.locval2bool = None
        self.edgeval2bool = None

    def _get_loc_var_name(self):
        """ Returns the variable used to represent the locations """
        return "loc"

    def _get_edge_var_name(self):
        """ Returns the variable used to represent the edge """
        return "edge"
 
    def next_clause(self,c):
        assert type(c) == list
        cprime = []
        for l in c:
            assert type(l) != list
            lprime = self._ts.helper.get_next_formula(self._ts.state_vars,l)
            cprime.append(lprime)
        return cprime

    def is_cnf(self, cnf_list):                
        assert type(cnf_list) == list
        for c in cnf_list:
            assert type(c) == list
            for l in c: 
                assert type(l) != list
                fop = l.node_type()

                if fop == op.NOT:
                    symbol = l.args()[0]
                else:
                    symbol = l
                fop = symbol.node_type()
                if not (fop == op.SYMBOL or
                        fop == op.LE or
                        fop == op.EQUALS):
                    return False
        return True

    def _convert(self):
        """ Create the transition system that encodes the input PWA.

        1. Creates the boolean encoding for the location variables
        2. Add the continuous variables
        3. Initial states
        4. Final states
        5. Location invariants
        6. Transition relation
        """
        self._ts = TransitionSystem(self.pysmt_env, self.helper)
        self._loc_enc = CounterEnc(self.pysmt_env)
        self.val2loc = {}
        self.loc2val = {}
        self.val2edge = {}
        self.edge2val = {}
        self.pysmtvars = []
        self.pysmtvars2index = {}

        self.locval2bool = {}
        self.edgeval2bool = {}

        # Encode the predicates directly in CNF to be able to use SATEX
        # WARNING: we do not encode the locations as CNF formula
        #          This is still sound for satex, the real issues are 
        #          theory predicates and their negation
        init_clauses = []
        final_clauses = []
        trans_clauses = []
        loc_invars_clauses = []

        # 1. Creates the boolean encoding for the location variables
        self._loc_enc.add_var(self._get_loc_var_name(),
                              # -1: the counter starts from 1
                              len(self.pwa_graph.nodes()) - 1) 
        # for v in self._loc_enc.get_counter_var(self._get_loc_var_name()):
        #     self._ts.add_var(v)
        #     self._ts.var_types[v] = BOOL

        loc_id = -1
        loc_vars = []
        for loc in self.pwa_graph.nodes():
            assert isinstance(loc, Qx)
            loc_id += 1
            self.val2loc[loc_id] = loc
            self.loc2val[loc] = loc_id

            newbvar = Symbol("_loc_bvar_%d" % loc_id, BOOL)
            self._ts.add_var(newbvar)
            self._ts.var_types[newbvar] = BOOL
            self.locval2bool[loc_id] = newbvar
            loc_vars.append(newbvar)
            # loc_invars_clauses.append(Iff(newbvar,
            #                               self._loc_enc.eq_val(self._get_loc_var_name(),
            #                                                    loc_id)))
        loc_invars_clauses.extend(self._only_one(loc_vars))

        # 2. Add the continuous variables
        index = -1
        for var_name in self.vs:
            index += 1
            str_var_name = str(var_name)
            pysmt_var = Symbol(str_var_name, REAL)
            self.pysmtvars.append(pysmt_var)
            self.pysmtvars2index[pysmt_var] = index
            self._ts.add_var(pysmt_var)
            self._ts.var_types[pysmt_var] = REAL

        # 3. Initial states
        init_clauses.extend(self._convert_IntervalCons(self.init_cons))
        init_clauses.extend(self._get_mod_part_set_enc(self.init_ps))

        # 4. Final states
        final_clauses.extend(self._convert_IntervalCons(self.final_cons))
        final_clauses.extend(self._get_mod_part_set_enc(self.final_ps))

        # 5. Location invariants
        # \bigwedge_{location in locations} (!location \/ location_invar)
        loc_vars = []
        for loc in self.pwa_graph.nodes():
            assert loc.dim == len(self.vs) and loc.xdim == len(self.vs)
            assert isinstance(loc.ival_constraints, constraints.IntervalCons)
            loc_invar = self._convert_IntervalCons(loc.ival_constraints)
            loc_enc = self._get_loc_enc(loc)
            for l in loc_invar:
                lp = list(l)
                lp.append(Not(loc_enc))
                loc_invars_clauses.append(lp)
            loc_vars.append(loc_enc)
        loc_invars_clauses.extend(self._only_one(loc_vars))

        # 6. Transition relation
        # \bigvee_{(loc, edge, loc') \in Edges} {
        #   (!edge or loc) &
        #   (!edge or loc')
        #   (!edge or loc_partition)
        #   (!edge or loc_partition')
        #   (!edge or edge_relation)
        # 
        # }
        self._loc_enc.add_var(self._get_edge_var_name(),
                              len(self.pwa_graph.all_edges()) - 1) 
        # for v in self._loc_enc.get_counter_var(self._get_edge_var_name()):
        #     self._ts.add_var(v)
        #     self._ts.var_types[v] = BOOL

        edge_id = -1
        edge_vars = []
        for edge in self.pwa_graph.all_edges():
            edge_id += 1
            self.val2edge[edge_id] = edge
            self.edge2val[edge] = edge_id

            newbvar = Symbol("_edge_bvar_%d" % edge_id, BOOL)
            self._ts.add_var(newbvar)
            self._ts.var_types[newbvar] = BOOL
            self.edgeval2bool[edge_id] = newbvar
            edge_vars.append(newbvar)
            # loc_invars_clauses.append(Iff(newbvar,
            #                               self._loc_enc.eq_val(self._get_edge_var_name(),
            #                                                    edge_id)))

            # Unpack the objects
            assert isinstance(edge, tuple) and len(edge) == 2
            src_part = self.pwa_graph.node_p(edge[0])
            assert isinstance(src_part, ModelPartition)
            dst_part = self.pwa_graph.node_p(edge[1])
            assert isinstance(dst_part, ModelPartition)
            edge_rel = self.pwa_graph.edge_m(edge)
            assert isinstance(dst_part, ModelPartition)

            # create the encoding
            src_loc_smt = self._get_loc_enc(edge[0])
            dst_loc_smt = self._ts.helper.get_next_formula(self._ts.state_vars,
                                                           self._get_loc_enc(edge[1]))

            rel_enc = self._get_relation_enc(edge_rel)
            src_part_smt = self._convert_partition(src_part)
            dst_part_smt = self._convert_partition(dst_part)

            edge_enc = self._get_edge_enc(edge)
            trans_clauses.append([Not(edge_enc), src_loc_smt])
            trans_clauses.append([Not(edge_enc), dst_loc_smt])

            for p in src_part_smt:
                assert type(p) == list
                newp = list(p)
                newp.append(Not(edge_enc))
                trans_clauses.append(newp)
            for p in dst_part_smt:
                assert type(p) == list
                pprime = self.next_clause(p)
                pprime.append(Not(edge_enc))
                trans_clauses.append(pprime)
            for r in rel_enc:
                trans_clauses.append([Not(edge_enc), r])

        loc_invars_clauses.extend(self._only_one(edge_vars))

        for c in loc_invars_clauses:
            cprime = self.next_clause(c)
            trans_clauses.append(cprime)
            init_clauses.append(c)

        self._ts.final = self.cnf_to_pysmt(final_clauses)
        self._ts.init = self.cnf_to_pysmt(init_clauses)
        self._ts.trans = self.cnf_to_pysmt(trans_clauses)
        
        # print self._ts.trans.serialize()
        # print self._ts.init.serialize()
        # print self._ts.final.serialize()
        # self._print_dot()

    def cnf_to_pysmt(self,cnf):
        # DISABLED
        # assert self.is_cnf(cnf)
        c_and = TRUE()
        for c in cnf:
            assert(type(c) == list)
            or_f = FALSE()
            for l in c:
                or_f = Or(or_f, l)
            c_and = And(c_and, or_f)
        return c_and

    def _get_relation_enc(self, edge_rel):
        """ Encode x' = Ax + b +- error
        as 
        Ax + b - error <= x' /\ x' <= Ax + b + error

        """
        assert len(self.pysmtvars) == len(edge_rel.A)
        assert len(self.pysmtvars) == len(edge_rel.b)
        assert len(self.pysmtvars) == len(edge_rel.error.l)
        assert len(self.pysmtvars) == len(edge_rel.error.h)
    
        rel_enc = []
        for i in range(len(self.pysmtvars) * 2):
            if (i < len(self.pysmtvars)):
                i_index = i
                e_e_smt = self.to_real(edge_rel.error.h[i])
            else:
                i_index = i - len(self.pysmtvars)
                e_e_smt = self.to_real(edge_rel.error.l[i_index])

            a_row = edge_rel.A[i_index]
            b_e_smt = self.to_real(edge_rel.b[i_index])

            current_var = self.pysmtvars[i_index]
            var_next = Helper.get_next_var(current_var,
                                           self.pysmt_env.formula_manager)

            # row - column product
            assert len(self.pysmtvars) == len(a_row)
            row_column = None
            for (var, a_el) in zip(self.pysmtvars, a_row):
                row_elem = Times(self.to_real(a_el), var)
                if row_column is None:
                    row_column = row_elem
                else:
                    row_column = Plus(row_column, row_elem)
            assert row_column is not None

            if (i < len(self.pysmtvars)):
                pred = LE(var_next, Plus(row_column, b_e_smt, e_e_smt))
            else:
                pred = LE(Plus(row_column, Plus(b_e_smt, e_e_smt)), var_next)

            rel_enc.append(pred)

        return rel_enc

    def _get_mod_part_set_enc(self, loc_set):
        """ Return a clause """
        enc_loc_set_smt = []
        locs_dis = []
        for model_partition in loc_set:
            (loc,mod_part_enc) = self._get_mod_part_enc(model_partition)
            locs_dis.append(loc)
            enc_loc_set_smt.extend(mod_part_enc)
        enc_loc_set_smt.append(locs_dis)
        return enc_loc_set_smt

    def _get_mod_part_enc(self, mod_part):
        assert isinstance(mod_part, ModelPartition)
        constraints = []
        loc = mod_part.ID
        assert isinstance(loc, Qx)
        loc_enc = self._get_loc_enc(loc)
        part_enc = self._convert_partition(mod_part)
        for p in part_enc:
            assert type(p) == list
            # loc -> partition
            p.append(Not(loc_enc))
            constraints.append(p)
        return (loc_enc,constraints)

    def _get_loc_enc(self, loc):
        """ Loc is a node in the pwa graph """
        loc_id = self.loc2val[loc]
        loc_enc = self._loc_enc.eq_val(self._get_loc_var_name(), loc_id)
        loc_enc = And(loc_enc,
                      self._loc_enc.get_mask(self._get_loc_var_name()))
        return self.locval2bool[loc_id]
        # return loc_enc

    def get_loc(self, model):
        for locval, newbvar in self.locval2bool.iteritems():
            if model[newbvar]:
                return self.val2loc[locval]
        return None

    def get_edge(self, model):
        for edgeval, newbvar in self.edgeval2bool.iteritems():
            if model[newbvar]:
                return self.val2edge[edgeval]
        return None


    def _get_edge_enc(self, edge):
        """ Loc is a node in the pwa graph """
        edge_id = self.edge2val[edge]
        edge_enc = self._loc_enc.eq_val(self._get_edge_var_name(), edge_id)
        edge_enc = And(edge_enc,
                       self._loc_enc.get_mask(self._get_edge_var_name()))
        return self.edgeval2bool[edge_id]
        # return edge_enc


    def _convert_IntervalCons(self, interval_cons):
        """ Convert a box constraint """
        assert len(interval_cons.l) == len(self.pysmtvars)
        assert len(interval_cons.h) == len(self.pysmtvars)
        constraints = []
        for (var, l, h) in zip(self.pysmtvars, interval_cons.l, interval_cons.h):
            try:
                l_val = self.to_real(l)
            except ErrorInfinity:
                pass
            else:
                constraints.append([LE(l_val, var)])
            try:
                h_val = self.to_real(h)
            except ErrorInfinity:
                pass
            else:
                constraints.append([LE(var, h_val)])
        return constraints

    def _convert_partition(self, partition):
        """ Each row of C  and d represents a constraint """
        assert isinstance(partition, ModelPartition)
        assert len(self.pysmtvars) > 0

        enc_list = []
        # generates the constraint for a row: c x <= d 
        for (c_row,d) in zip(partition.C, partition.d):
            lin_comb = None
            for (var, c) in zip(self.pysmtvars, c_row):
                c_smt = self.to_real(c)
                lhs_smt = Times(var, c_smt)
                if lin_comb is None:
                    lin_comb = lhs_smt
                else:
                    lin_comb = Plus(lin_comb, lhs_smt)
            d_smt = self.to_real(d)
            pred = LE(lin_comb, d_smt)
            enc_list.append([pred])
        return enc_list


    def get_index(self, var):
        if var in self.pysmtvars2index:
            return self.pysmtvars2index[var]
        else:
            return -1
        
    def get_ts(self):
        """ Return the transition system """

        if self._ts is None:
            self._convert()
        assert self._ts is not None
        return self._ts

    
    def to_real(self, n):
        # num_str = str(n)
        # return Real(Fraction(num_str))

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

    def to_float(self, real):
        return float(Fraction(str(real)))

    def _print_dot(self):
        locs = set()
        locs_string = {}
        is_initial = set()
        is_final = set()

        edges = []

        for loc in self.pwa_graph.nodes():
            loc_id = self.loc2val[loc]
            assert loc_id not in locs
            locs.add(loc_id)
            locs_string[loc_id] = str(loc)
            
        for mp in self.init_ps:
            loc_id = self.loc2val[mp.ID]
            assert loc_id in locs
            is_initial.add(loc_id)
            
        for mp in self.final_ps:
            loc_id = self.loc2val[mp.ID]
            assert loc_id in locs
            is_final.add(loc_id)

        for edge in self.pwa_graph.all_edges():
            src_loc_id = self.loc2val[edge[0]]
            dst_loc_id = self.loc2val[edge[1]]
            assert src_loc_id in locs
            assert dst_loc_id in locs

            edges.append((src_loc_id, dst_loc_id))

        with open("app.dot", "w") as f:
            f.write("digraph{\n")

            for loc_id in locs:
                if loc_id in is_initial:
                    shape = "rectangle"
                elif loc_id in is_final:
                    shape = "doublecircle"
                else:
                    shape = "circle"
                f.write("node [shape=%s,label=\"%s\"] %d;\n" % (shape,
                                                              locs_string[loc_id],
                                                              loc_id))

            for (src_loc_id, dst_loc_id) in edges:
                f.write("%d -> %d;\n" % (src_loc_id, dst_loc_id))

            f.write("}")
            f.close()

    def _only_one(self, vars):
        res = [vars]

        remaining = list(vars)
        while len(remaining) > 1:
            v1 = remaining.pop()
            assert v1 is not None
            for v2 in remaining:
                assert v2 is not None
                res.append([Not(v1), Not(v2)])
        return res

        
        
