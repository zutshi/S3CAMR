import itertools as it
# full precision
PREC = '0.5'


def float2str(n):
    """float2str

    Parameters
    ----------
    n : floating point number

    Returns
    -------
    String notation truncated to PREC

    Notes
    ------
    Returns empty string '' if after truncation the float is 0.0
    """
    return '{n:{p}f}'.format(p=PREC, n=n).rstrip('0').rstrip('.')


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
    cstr = float2str(c)
    if cstr == '0' or cstr == '-0':
        cxstr = ''
    elif cstr == '1':
        cxstr = x
    elif cstr == '-1':
        cxstr = '-' + x
    else:
        cxstr = '{}*{}'.format(cstr, x)
    return cxstr


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
    b1 = (b, str(1.0))
    cx = it.chain(it.izip(coeffs, xs), [b1])
    return ' + '.join((filter(None, (cx2str(c, x) for c, x in cx))))
