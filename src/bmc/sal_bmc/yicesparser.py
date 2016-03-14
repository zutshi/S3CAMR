'''parses yices output'''

import pyparsing as pp
import argparse
from fractions import Fraction as FR

import fileops as fops

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

rational_integer = rational | signed_integer


def Assignment(s):
    return ' '.join(s)


def Step(s):
    #step_num = s[0]
    s = '({})'.format(', '.join(s[1:]))
    return s


def extract_label(s):
    return s[1]


def parse_trace(trace_data):
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

    assignment = ident + EQUAL + rational_integer
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


def main():
    usage = '%(prog)s <filename>'
    parser = argparse.ArgumentParser(description='demo pwa', usage=usage)
    parser.add_argument('trace_file', default=None, type=str)
    args = parser.parse_args()
    parsed_trace = parse_trace(fops.get_data(args.trace_file))

    for i in parsed_trace:
        print i

if __name__ == '__main__':
    main()