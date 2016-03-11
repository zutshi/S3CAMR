import numpy as np
import os

import saltrans as slt
import fileops as fops
import utils as U
import err

np.set_printoptions(suppress=True, precision=2)

#SAL_PATH = '''../../sal-3.3/bin/sal-inf-bmc'''
SAL_INF_BMC = '''/bin/sal-inf-bmc'''


# Must separate the arguements. i.e., -v 3 should be given as ['-v', '3']
# This can be avoided by using shell=True, but that is a security risk
def sal_run_cmd(sal_path, depth, module_name, prop_name, verbosity=3):
    return [
        sal_path,
        '-v', str(verbosity),
        '-d', str(depth),
        '{}.sal'.format(module_name),
        prop_name
    ]


class BMC(object):
    def __init__(self, nd, pwa_model, init_state, safety_prop, module_name):
        self.sal_trans_sys = self.__class__.sal_module(nd,
                                                       pwa_model,
                                                       init_state,
                                                       safety_prop,
                                                       module_name)

        self.prop_name = 'safety'
        self.module_name = module_name
        return

    @staticmethod
    def sal_module(nd, pwa_model, init_set, safety_prop, module_name):
        sal_trans_sys = slt.SALTransSys(module_name, nd, init_set, safety_prop)

        for sub_model in pwa_model:
            g = slt.Guard(sub_model.p.C, sub_model.p.d)
            r = slt.Reset(sub_model.m.A, sub_model.m.b)
            t = slt.Transition(g, r)
            sal_trans_sys.add_transition(t)
        return sal_trans_sys

    def check(self, depth):
        try:
            sal_path_ = os.environ['SAL'] + SAL_INF_BMC
        except KeyError:
            err.error("SAL environment variable is not defined. It\n"
                      "should point to sal's top-level directory")
            #raise KeyError

        sal_path = fops.sanitize_path(sal_path_)
        verbosity = 3
        sal_cmd = sal_run_cmd(
                    sal_path,
                    depth,
                    self.module_name,
                    self.prop_name,
                    verbosity)
        if __debug__:
            print sal_cmd
            U.strict_call(['echo'] + sal_cmd)
        U.strict_call(sal_cmd)

    def dump(self):
        sal_file = self.module_name + '.sal'
        fops.write_data(sal_file, str(self.sal_trans_sys))
        return
