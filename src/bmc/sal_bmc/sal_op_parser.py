from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''parses yices output'''

import pyparsing as pp
import argparse
from fractions import Fraction as FR

import fileops as fops
import err

import functools

from bmc.bmc_spec import TraceSimple

# global tokens
SEMI = pp.Literal(";").suppress()
COLON = pp.Literal(":").suppress()
COMMA = pp.Literal(",").suppress()
PLUS = pp.Literal("+")
MINUS = pp.Literal("-")
UNDERSCORE = pp.Literal('_')
EQUAL = pp.Literal('=')
#LPAREN = pp.Literal("(").suppress()
#RPAREN = pp.Literal(")").suppress()


def Rational(s):
    pq_str = ''.join(s)
    f = float(FR(pq_str))
    return str(f)


def Float(s):
    return ''.join(s)


def SignedInteger(s):
    return ''.join(s)

integer = pp.Word(pp.nums)
signed_integer = pp.Optional(PLUS | MINUS) + integer
alpha = pp.alphas
#IDENT = pp.Word(pp.srange("[a-zA-Z_]"), pp.srange("[a-zA-Z0-9_]"))
ident = pp.Word(pp.alphanums+"_")
rational = signed_integer + pp.Literal('/') + integer

rational.setParseAction(Rational)
signed_integer.setParseAction(SignedInteger)
floats = signed_integer + pp.Literal('.') + integer
floats.setParseAction(Float)

value = rational | signed_integer | ident


class Assignment(object):
    def __init__(self, s):
        self.s = ' '.join(s)
        self.lhs = s[0]
        try:
            self.rhs = float(s[2])
        except ValueError:
            self.rhs = s[2]

    def __str__(self):
        return self.s


class Step(object):
    def __init__(self, s):
        self.s = s
        self.num = s[0]
        assignments = s[1:-1]
        self.assignments = {a.lhs: a.rhs for a in assignments}
        # transition id string
        self.tid = s[-1]

    def __str__(self):
        return '({})'.format(', '.join(str(i) for i in self.s[1:]))


def Trace(vs, s):
    trace_with_NOP = s[:-1]
    exec_time = s[-1]

    # remove NOPs
    trace = [step for step in trace_with_NOP if step.tid != 'NOP']

    return TraceSimple(trace, vs)


def extract_label(s):
    return s[1]


def parse_ce(trace_data, vs):
    # lexer rules
    SEP = pp.Keyword('------------------------').suppress()
    STEP = pp.Keyword('Step').suppress()
    HDR_ = (pp.Keyword('Counterexample:') +
            pp.Keyword('========================') +
            pp.Keyword('Path') +
            pp.Keyword('========================')).suppress()
    # ignore the version information before the HDR_
    HDR = pp.SkipTo(HDR_, True).suppress()
    FOOTER = pp.Keyword('total execution time:') + floats + pp.Keyword('secs')
    FOOTER.setParseAction(''.join)
    EOF = pp.StringEnd()
    LABEL = pp.Keyword('label')

    # Grammar
    #LABEL = pp.Keyword('label').suppress()
    sva = pp.Keyword('--- System Variables (assignments) ---').suppress()
    # XXX: SAL's default monitor?
    #bapc = (pp.Keyword('ba-pc!1') + EQUAL + integer).suppress()
    step_hdr = STEP + integer + COLON

    assignment = ident + EQUAL + value
    assignment.setParseAction(Assignment)
    label = LABEL.suppress() + ident
    ti = SEP + pp.SkipTo(label, False) + label + pp.SkipTo(SEP, True)
    ti.setParseAction(extract_label)
    #step = step_hdr + sva + bapc + pp.OneOrMore(assignment) + pp.Optional(ti)
    step = step_hdr + sva + pp.OneOrMore(assignment) + pp.Optional(ti, default='')
    step.setParseAction(Step)

    #step.setParseAction(Step)
    trace = (HDR + pp.OneOrMore(step) +
             pp.Optional(FOOTER, default='')
             )
#                 ) +
#             pp.SkipTo(EOF, True))
    trace.setParseAction(functools.partial(Trace, vs))

    parsed = trace.parseString(trace_data, parseAll=True)
    return parsed[0]


def parse_trace(trace_data, vs):
    """pre_process
    Quick check if SAL has failed to find any counter example.
    """

    #TODO: quick fix
    import re
    trace_data = re.sub('trace.*\n', '', trace_data)

    yices_failed = 'The context is unsat. No model.'
    sal_failed = 'no counterexample between depths:'

    if yices_failed in trace_data and sal_failed in trace_data:
        return None
    elif yices_failed in trace_data or sal_failed in trace_data:
        raise err.Fatal('unexpected trace data')
    else:
        return parse_ce(trace_data, vs)


def main():
    usage = '%(prog)s <filename>'
    parser = argparse.ArgumentParser(description='demo pwa', usage=usage)
    parser.add_argument('trace_file', default=None, type=str)
    args = parser.parse_args()
    parsed_trace = parse_trace(fops.get_data(args.trace_file))

    if parsed_trace is None:
        print('No CE found!')
    else:
        #print(parsed_trace)
        for ass in parsed_trace:
            print(ass)

if __name__ == '__main__':
    main()
