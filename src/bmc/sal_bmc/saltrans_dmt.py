from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


'''
Creates a SAL transition system
'''

from . import saltrans


class SALTransSysDMT(saltrans.SALTransSys):

    def __init__(self, dts, *args):
        super(SALTransSysDMT, self).__init__(*args)
        self.dts = dts
        self.dt_set_repr = '{{{}}}'.format(
                ', '.join([str(dt) for dt in self.dts]))
        return

    @property
    def always_true_transition(self):
        s = super(SALTransSysDMT, self).always_true_transition
        #t = ";\nt' IN {}".format(self.dt_set_repr)
        t = ";\nt' = t"
        return s + t

    @property
    def decls(self):
        decls = super(SALTransSysDMT, self).decls
        t_decl = ',\n\tt:REAL'
        return decls + t_decl

    @property
    def init_set(self):
        init_set = super(SALTransSysDMT, self).init_set
        t_init = ';\n\tt IN ' + self.dt_set_repr
        return init_set + t_init

    # sal description
    @property
    def trans(self):
        ts = '[]\n'.join(('{}'.format(i) for i in self.transitions))
        s = '{}'.format(ts)
        return s


class Guard(saltrans.Guard):
    pass


class Reset(saltrans.Reset):
    pass


class Transition(saltrans.Transition):

    def __init__(self, dt, dts, *args):
        super(Transition, self).__init__(*args)
        self.dt = dt
        self.dt_set_repr = '{{{}}}'.format(
                ', '.join([str(dt_) for dt_ in dts]))
        return

    def __str__(self):
        guard_t = 't = {}'.format(self.dt)
        reset_t = "t' IN " + self.dt_set_repr
        s = '{}:\n{} AND {} -->\n{};\n{}\n'.format(self.name, self.g, guard_t, self.r, reset_t)
        return s
