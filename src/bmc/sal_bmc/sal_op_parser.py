'''parses yices output'''

import pyparsing as pp
import argparse
from fractions import Fraction as FR

import fileops as fops
import err

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


integer = pp.Word(pp.nums)
signed_integer = pp.Optional(PLUS | MINUS) + integer
alpha = pp.alphas
#IDENT = pp.Word(pp.srange("[a-zA-Z_]"), pp.srange("[a-zA-Z0-9_]"))
ident = pp.Word(pp.alphanums+"_")
rational = signed_integer + pp.Literal('/') + integer

rational.setParseAction(Rational)
#signed_integer.setParseAction(SignedInteger)

value = rational | signed_integer | ident


def Assignment(s):
    return ' '.join(s)


def Step(s):
    #step_num = s[0]
    s = '({})'.format(', '.join(s[1:]))
    return s


def extract_label(s):
    return s[1]


def parse_ce(trace_data):
    # lexer rules
    SEP = pp.Keyword('------------------------').suppress()
    STEP = pp.Keyword('Step').suppress()
    HDR_ = (pp.Keyword('Counterexample:') +
            pp.Keyword('========================') +
            pp.Keyword('Path') +
            pp.Keyword('========================')).suppress()
    # ignore the version information before the HDR_
    HDR = pp.SkipTo(HDR_, True)
    FOOTER = pp.Keyword('total execution time:')
    EOF = pp.StringEnd()
    LABEL = pp.Keyword('label')

    # Grammar
    #LABEL = pp.Keyword('label').suppress()
    sva = pp.Keyword('--- System Variables (assignments) ---').suppress()
    # XXX: not very clear what this denotes
    bapc = (pp.Keyword('ba-pc!1') + EQUAL + integer).suppress()
    step_hdr = STEP + integer + COLON

    assignment = ident + EQUAL + value
    assignment.setParseAction(Assignment)
    label = LABEL.suppress() + ident
    ti = SEP + pp.SkipTo(label, False) + label + pp.SkipTo(SEP, True)
    ti.setParseAction(extract_label)
    step = step_hdr + sva + bapc + pp.OneOrMore(assignment) + pp.Optional(ti)
    step.setParseAction(Step)

    #step.setParseAction(Step)
    trace = HDR + pp.OneOrMore(step) + pp.Optional(FOOTER) + pp.SkipTo(EOF, True)

    parsed = trace.parseString(trace_data, parseAll=True)
    return parsed


#TODO: make a class for parsed_trace
def cleanup_parsed_trace(parsed_trace):
    return '\n'.join(parsed_trace)


def parse_trace(trace_data):
    """pre_process
    Quick check if SAL has failed to find any counter example.
    """
    yices_failed = 'The context is unsat. No model.'
    sal_failed = 'no counterexample between depths:'

    if yices_failed in trace_data and sal_failed in trace_data:
        return None
    elif yices_failed in trace_data or sal_failed in trace_data:
        raise err.Fatal('unexpected trace data')
    else:
        return cleanup_parsed_trace(parse_ce(trace_data))


def main():
    usage = '%(prog)s <filename>'
    parser = argparse.ArgumentParser(description='demo pwa', usage=usage)
    parser.add_argument('trace_file', default=None, type=str)
    args = parser.parse_args()
    parsed_trace = parse_trace(fops.get_data(args.trace_file))

    if parsed_trace is None:
        print 'No CE found!'
    else:
        print parsed_trace

if __name__ == '__main__':
    main()
