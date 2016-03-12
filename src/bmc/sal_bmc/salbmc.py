import numpy as np
import os

import saltrans as slt
import saltrans_dmt as slt_dmt
import fileops as fops
import utils as U
import err

np.set_printoptions(suppress=True, precision=2)

#SAL_PATH = '''../../sal-3.3/bin/sal-inf-bmc'''
SAL_INF_BMC = '''/bin/sal-inf-bmc'''


class SALBMCError(Exception):
    pass


# Must separate the arguements. i.e., -v 3 should be given as ['-v', '3']
# This can be avoided by using shell=True, but that is a security risk
def sal_run_cmd(sal_path, depth, module_name, prop_name, verbosity=3, iterative=False):
    cmd = [
        sal_path,
        '-v', str(verbosity),
        '-d', str(depth),
        '-s', 'yices2',
        '{}.sal'.format(module_name),
        prop_name
    ]
    if iterative:
        cmd.append('-it')
    return cmd



class BMC(object):
    def __init__(self, nd, pwa_model, init_state, safety_prop,
                 module_name, model_type):
        if model_type == 'dft':
            self.sal_trans_sys = BMC.sal_module_dft(
                    nd, pwa_model, init_state, safety_prop, module_name)
        elif model_type == 'dmt':
            dts = pwa_model.keys()
            self.sal_trans_sys = BMC.sal_module_dmt(
                    dts, nd, pwa_model, init_state, safety_prop, module_name)
        elif model_type == 'ct':
            raise NotImplementedError
        else:
            raise SALBMCError('unknown model type')

        self.prop_name = 'safety'
        self.module_name = module_name
        return

    @staticmethod
    def sal_module_dmt(dts, nd, pwa_models, init_set, safety_prop, module_name):
        sal_trans_sys = slt_dmt.SALTransSysDMT(dts, module_name, nd, init_set, safety_prop)
        for dt, pwa_model in pwa_models.iteritems():
            for sub_model in pwa_model:
                g = slt_dmt.Guard(sub_model.p.C, sub_model.p.d)
                r = slt_dmt.Reset(sub_model.m.A, sub_model.m.b)
                t = slt_dmt.Transition(dt, dts, g, r)
                sal_trans_sys.add_transition(t)
        return sal_trans_sys

    @staticmethod
    def sal_module_dft(nd, pwa_model, init_set, safety_prop, module_name):
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
