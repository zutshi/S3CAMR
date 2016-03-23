import numpy as np
import constraints as cons


class PWAError(Exception):
    pass


class PWA(object):
    def __init__(self):
        self.nlocs = 0
        self.sub_models = {}
        self.idx = 0
        return

    def add_sub_model(self, guard, reset):
        part_id = self.idx
        self.sub_models[part_id] = SubModel(guard, reset)
        self.idx += 1
        return

    def add(self, sub_model):
        part_id = self.idx
        self.sub_models[part_id] = sub_model
        self.idx += 1
        return

    def get_sub_model(self, part_id):
        return self.sub_models[part_id]

    # Make the class iterable
    def __iter__(self):
        return self.sub_models.itervalues()

    #def next(self):
    #    return self.sub_models.itervalues().next()

    def __repr__(self):
        return repr(self.sub_models)

    def __str__(self):
        s = [str(i) for i in self]
        return '{1}{0}{1}'.format('='*20, '\n').join(s)


    # returns the first sub_model whose parition the point x belongs
    # to
    # TODO: Brute force search, very inefficient
    def find_sub_model(self, x):
        for sub_model in self.sub_models.itervalues():
            if poly_sat(sub_model.p, x):
                return sub_model
        raise PWAError('no appropriate submodel found')

    def predict(self, x):
        try:
            sub_model = self.find_sub_model(x)
        except PWAError:
            return x

        m = sub_model.m
        return np.dot(m.A, x) + m.b


def poly_sat(poly, x):
    """poly_sat

    Parameters
    ----------
    poly : polytope
    x : vector

    Returns True if x is a member of polytope p
    """
#     print poly.C
#     print poly.d
#     print x
    return np.all(np.dot(poly.C, x) <= poly.d)


def compute_part_id(guard):
    return


def sub_model_helper(A, b, C, d, e=None):
    model = Model(A, b, e)
    partition = Partition(C, d)
    return SubModel(partition, model)


class SubModel(object):
    def __init__(self, p, m):
        '''
        loc: p(x) => x' = m(x)
        '''
        self.p = p
        self.m = m
        return

    def __repr__(self):
        s = '({},{})'.format(self.p, self.m)
        return s

    def __str__(self):
        s = 'SubModel ->(\n{},\n{})'.format(self.p, self.m)
        return s


class Partition(object):
    def __init__(self, C, d):
        '''
        Cx <= d
        '''
        self.C = C
        self.d = d
        return

    def __repr__(self):
        s = '({},{})'.format(self.C, self.d)
        return s

    def __str__(self):
        s = 'Pi ->(\n{},\n{})'.format(self.C, self.d)
        return s


class Model(object):
    def __init__(self, A, b, e=None):
        '''
        x' = Ax + b
        '''
        self.A = A
        self.b = b
        self._error = e
        return

    @property
    def error(self):
        #if self._error is None:
        #    raise AttributeError('error has not been set yet!')
        return self._error

    @error.setter
    def error(self, val):
        assert(isinstance(val) == cons.IntervalCons)
        self._error = val

    def __repr__(self):
        s = '({},{},{})'.format(self.A, self.b, self.e)
        return s

    def __str__(self):
        s = 'Mi - >(\n{},\n{}+-{})'.format(self.A, self.b, self.e)
        return s
