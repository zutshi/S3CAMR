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


def compute_part_id(guard):
    return


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
        Ax + b <= 0
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
    def __init__(self, A, b):
        '''
        x' = Ax + b
        '''
        self.A = A
        self.b = b
        return

    def __repr__(self):
        s = '({},{})'.format(self.A, self.b)
        return s

    def __str__(self):
        s = 'Mi - >(\n{},\n{})'.format(self.A, self.b)
        return s
