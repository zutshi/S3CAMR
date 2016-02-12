class PWA(object):
    def __init__(self):
        self.nlocs = 0
        self.sys = {}
        return

    def add_region(self, region_id, guard, reset):
        self.sys[region_id] = Region(guard, reset)
        return

    def get_region(self, region_id):
        return self.sys[region_id]


def compute_region_id(guard):
    raise NotImplementedError
    return


class Region(object):
    def __init__(self, g, r):
        '''
        loc: g(x) => x' = r(x)
        '''
        self.g = g
        self.r = r
        return


class Guard(object):
    def __init__(self, A, b):
        '''
        Ax + b <= 0
        '''
        self.A = A
        self.b = b
        return


class Reset(object):
    def __init__(self, A, b):
        '''
        x' = Ax + b
        '''
        self.A = A
        self.b = b
        return
