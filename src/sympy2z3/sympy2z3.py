from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import operator

import sympy as sym
import z3

import utils as U
from utils import print_function


def sympy2z3(sympy_exprs):
    """
    convert a sympy expression to a z3 expression. This returns
    (z3_vars, z3_expression)

    Parameters
    ----------
    sympy_exprs : iterable of expressions

    Returns
    -------

    Notes
    ------
    """
    assert(isinstance(sympy_exprs, collections.Iterable))

    z3_exprs = []
    sym2Z3_varmap = {}

    for expr in sympy_exprs:
        assert(isinstance(expr, sym.Expr))
        #print(expr)
        sympy_vars = expr.free_symbols

        for v in sympy_vars:
            U.dict_unique_add(sym2Z3_varmap, v, z3.Real(str(v)))
        #sym2Z3_varmap = {v: z3.Real(str(v)) for v in sympy_vars}

#         sym2Z3_varmap = {v: z3.Real('x{}'.format(idx))
#                          for idx, v in enumerate(sympy_vars)}

        t = Sympy2z3(sym2Z3_varmap)
        z3_expr = t.visit(expr)
        z3_exprs.append(z3_expr)
    return sym2Z3_varmap, z3_exprs


class Sympy2z3(object):

    """
    A node visitor base class that walks the abstract syntax tree and calls a
    visitor function for every node found.  This function may return a value
    which is forwarded by the `visit` method.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `visit` method.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.

    Don't use the `NodeVisitor` if you want to apply changes to nodes during
    traversing.  For this a special visitor exists (`NodeTransformer`) that
    allows modifications.
    """

    def __init__(self, sym2Z3_varmap):

        #assert(isinstance(sympy_vars, collections.Iterable))

        #self.sym2Z3_varmap = {v: z3.Real('x{}'.format(idx))
        #                      for idx, v in enumerate(sympy_vars)}
        self.sym2Z3_varmap = sym2Z3_varmap
        return

    def visit(self, node):
        """Visit a node."""
        class_str = str(node.__class__).strip("<>'")
        class_name = class_str[class_str.rfind('.')+1:]
        method = 'visit_' + class_name
        #print(method)
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        #print(node)
        raise RuntimeError
#         for field, value in iter_fields(node):
#             if isinstance(value, list):
#                 for item in value:
#                     if isinstance(item, AST):
#                         self.visit(item)
#             elif isinstance(value, AST):
#                 self.visit(value)

    def visit_Symbol(self, node):
        assert(isinstance(node, sym.Symbol))
        return self.sym2Z3_varmap[node]

    # Does not get triggered, instead integer, float, etc gets called
    def visit_Number(self, node):
        assert(isinstance(node, sym.Number))
        return float(node)

    def visit_Integer(self, node):
        assert(isinstance(node, sym.Integer))
        return float(node)

    def visit_Float(self, node):
        assert(isinstance(node, sym.Float))
        return float(node)

    def visit_NegativeOne(self, node):
        return float(-1)

    def visit_Zero(self, node):
        return float(0)

    def visit_Mul(self, node):
        assert(isinstance(node, sym.Mul))
        visited_terms = (self.visit(terms) for terms in node.args)
        return reduce(operator.mul, visited_terms)

    def visit_Add(self, node):
        assert(isinstance(node, sym.Add))
        visited_terms = (self.visit(terms) for terms in node.args)
        return reduce(operator.add, visited_terms)

    def visit_Pow(self, node):
        assert(isinstance(node, sym.Pow))
        return self.visit(node.args[0]) ** self.visit(node.args[1])

    def visit_LessThan(self, node):
        assert(isinstance(node, sym.LessThan))
        return self.visit(node.args[0]) <= self.visit(node.args[1])
