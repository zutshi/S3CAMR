
# full precision
PREC = '0.4'


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


def linexpr2str(coeffs, xs):
    """linexpr2str

    Parameters
    ----------
    coeffs : c0, c1, c2, ...
    xs : x0, x1, x2, ...

    Returns
    -------
    c0*x0 + c1*x1 + c2*x2 + ...

    Notes
    ------
    Cleans up the expression whenver ci = 0 or +-1
    """
    return ' + '.join((filter(None, (cx2str(c, x) for c, x in zip(coeffs, xs)))))
