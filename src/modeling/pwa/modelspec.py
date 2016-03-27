import abc

import numpy as np
import constraints as cons
from utils import poly_sat


class ModelError(Exception):
    pass


class ModelSpec():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add(self, sub_model):
        raise NotImplementedError

#     @abc.abstractmethod
#     def get_sub_model(self, part_id):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def find_sub_model(self, x):
#         raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class SubModelSpec():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sat(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError


# TODO: Merge with the abstract base class

class ModelGeneric(ModelSpec):
    def __init__(self):
        self.nlocs = 0
        self.sub_models = {}
        self.idx = 0
        return

    def add(self, sub_model):
        part_id = self.idx
        self.sub_models[part_id] = sub_model
        self.idx += 1
        return

    #def get_sub_model(self, part_id):
    #    return self.sub_models[part_id]

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
            if sub_model.sat(x):
                return sub_model
        raise ModelError('no appropriate submodel found')

    def predict(self, x):
        try:
            sub_model = self.find_sub_model(x)
        except ModelError:
            return x

        return sub_model.predict(x)


class Partition(object):
    def __init__(self, C, d, part_id):
        '''
        Cx <= d
        '''
        self.C = C
        self.d = d
        self.pid = part_id
        return

    def __repr__(self):
        s = '({},{})'.format(self.C, self.d)
        return s

    def __str__(self):
        s = 'Pi ->(\n{},\n{})'.format(self.C, self.d)
        return s


class DiscreteAffineMap(object):
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


class PartitionedDiscreteAffineModel(SubModelSpec):
    def __init__(self, p, m):
        '''
        loc: p(x) => x' = m(x)
        '''
        self.p = p
        self.m = m
        return

    def sat(self, x):
        return poly_sat(self.p, x)

    def predict(self, x):
        m = self.m
        return np.dot(m.A, x) + m.b

    def __repr__(self):
        s = '({},{})'.format(self.p, self.m)
        return s

    def __str__(self):
        s = 'SubModel ->(\n{},\n{})'.format(self.p, self.m)
        return s
