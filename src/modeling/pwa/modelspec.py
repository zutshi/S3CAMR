import abc

import numpy as np
import constraints as cons
from utils import poly_sat


class ModelError(Exception):
    pass


class ModelSpec(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def add(self, sub_model):
        return

    @abc.abstractmethod
    def find_sub_model(self, x):
        return

    @abc.abstractmethod
    def predict(self, x):
        return

    # Make the class iterable
    @abc.abstractmethod
    def __iter__(self):
        return

    @abc.abstractmethod
    def __repr__(self):
        return

    @abc.abstractmethod
    def __str__(self):
        return


class SubModelSpec():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sat(self, x):
        return

    @abc.abstractmethod
    def predict(self, x):
        return


class Partition(object):
    def __init__(self, C, d, part_id):
        '''
        Cx <= d
        '''
        self.C = C
        self.d = d
        self.ID = part_id
        return

    def __repr__(self):
        s = '({},{})'.format(self.C, self.d)
        return s

    def __str__(self):
        s = 'Pi ->(\n{},\n{})'.format(self.C, self.d)
        return s


class DiscreteAffineMap(object):
    def __init__(self, A, b, e):
        '''
        x' = Ax + b +- error
        '''
        self.A = A
        self.b = b
        assert(isinstance(e, cons.IntervalCons))
        self.error = e
        return

    def __repr__(self):
        s = '({},{},{})'.format(self.A, self.b, self.error)
        return s

    def __str__(self):
        s = 'Mi - >(\n{},\n{} + [{}, {}])'.format(
                self.A, self.b, self.error.l, self.error.h)
        return s


class PartitionedDiscreteAffineModel(SubModelSpec):
    def __init__(self, p, m):
        '''
        loc: p(x) => x' = m(x)
        '''
        self.p = p
        self.m = m
        self.ID = p.ID
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


class ModelGeneric(ModelSpec):
    def __init__(self):
        self.nlocs = 0
        self.sub_models = {}
        self.idx = 0
        #self.relation_ids = set()
        return

    def add(self, sub_model):
        part_id = self.idx
        self.sub_models[part_id] = sub_model
        self.idx += 1
        #self.relation_ids.add((sub_model.ID))

    #def get_sub_model(self, part_id):
    #    return self.sub_models[part_id]

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
            if __debug__:
                print('No sub-model found')
            return x

        return sub_model.predict(x)

    # Make the class iterable
    def __iter__(self):
        return self.sub_models.itervalues()

    def __repr__(self):
        return repr(self.sub_models)

    def __str__(self):
        s = [str(i) for i in self]
        return '{1}{0}{1}'.format('='*20, '\n').join(s)
