from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools as it
#import utils as U
#import err


class Expr2Str(object):
    PREC = None

    @staticmethod
    def set_prec(prec):
        assert(type(prec) == int)
        prec = '.' + str(prec)
        Expr2Str.PREC = prec

    @staticmethod
    def float2str(n):
        """float2str

        Parameters
        ----------
        n : floating point number

        Returns
        -------
        String notation truncated to PREC
        """
        assert(Expr2Str.PREC is not None)
        return '{n:{p}f}'.format(p=Expr2Str.PREC, n=n).rstrip('0').rstrip('.')

    @staticmethod
    def cx2str(c, x):
        """cx2str

        Parameters
        ----------
        c : coefficient
        x : varibale string

        Returns
        -------
        String notation, e.g. '2.345*x'

        Notes
        ------
        Returns ''

        """
        cstr = Expr2Str.float2str(c)
        if cstr == '0' or cstr == '-0':
            cxstr = ''
        elif cstr == '1':
            cxstr = x
        elif cstr == '-1':
            cxstr = '-' + x
        else:
            cxstr = '{}*{}'.format(cstr, x)
        return cxstr

    @staticmethod
    def linexpr2str(xs, coeffs, b):
        """linexpr2str
        coeffs .* xs + b
        Parameters
        ----------
        xs : x0, x1, x2, ...
        coeffs : c0, c1, c2, ...
        b : float

        Returns
        -------
        c0*x0 + c1*x1 + c2*x2 + ...

        Notes
        ------
        Cleans up the expression whenever ci = 0 or +-1
        """
        b1 = (b, '1')
        cx = it.chain(it.izip(coeffs, xs), [b1])
        ret = ' + '.join((filter(None, (Expr2Str.cx2str(c, x) for c, x in cx))))
        # make sure an empty string is not returned. This can happen if
        # the linexpr evaluates to a 0
        if not ret:
            ret = '0'
        return ret
