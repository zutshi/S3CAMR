import os

import saltrans as slt_dft
import saltrans_rel as slt_rel
import saltrans_dmt as slt_dmt
from ..bmc_spec import BMCSpec, InvarStatus
import sal_op_parser

import fileops as fops
import utils as U
import err

SAL_PATH = 'SAL_PATH'
SAL_INF_BMC = '''/bin/sal-inf-bmc'''


class SALBMCError(Exception):
    pass


# Must separate the arguements. i.e., -v 3 should be given as ['-v', '3']
# This can be avoided by using shell=True, but that is a security risk
def sal_run_cmd(sal_path, depth, module_name, prop_name, yices=2, verbosity=3, iterative=False):
    cmd = [
        sal_path,
        '-v', str(verbosity),
        '-d', str(depth),
        '{}.sal'.format(module_name),
        prop_name
    ]

    if yices == 2:
        cmd.extend(['-s', 'yices2'])

    if iterative:
        cmd.append('-it')

    print ' '.join(cmd)

    return cmd


class BMC(BMCSpec):
    def __init__(self, vs, pwa_model, init_state, safety_prop,
                 module_name, model_type):
        """__init__

        Parameters
        ----------
        vs : list of variables. Order is important.
        pwa_model : 
        init_state : 
        safety_prop : 
        module_name : 
        model_type : 

        Returns
        -------

        Notes
        ------
        """
        if model_type == 'dft':
            self.sal_trans_sys = BMC.sal_module_dft(
                    vs, pwa_model, init_state, safety_prop, module_name)
        elif model_type == 'dmt':
            dts = pwa_model.keys()
            self.sal_trans_sys = BMC.sal_module_dmt(
                    dts, vs, pwa_model, init_state, safety_prop, module_name)
        elif model_type == 'ct':
            raise NotImplementedError
        elif model_type == 'rel':
            self.sal_trans_sys = BMC.sal_module_rel(
                    vs, pwa_model, init_state, safety_prop, module_name)
        else:
            raise SALBMCError('unknown model type')

        self.prop_name = 'safety'
        self.module_name = module_name
        self.trace = None
        return

    @staticmethod
    def sal_module_rel(vs, pwa_model, init_set, safety_prop, module_name):
        sal_trans_sys = slt_rel.SALTransSysRel(module_name, vs, init_set, safety_prop)

        sal_trans_sys.add_locations(pwa_model.relation_ids)
        for idx, sub_model in enumerate(pwa_model):
            l1 = sal_trans_sys.get_loc_id(sub_model.p1.pid)
            l2 = sal_trans_sys.get_loc_id(sub_model.p2.pid)
            g = slt_rel.Guard(l1, sub_model.p1.C, sub_model.p1.d)
            r = slt_rel.Reset(l2, sub_model.m.A, sub_model.m.b, sub_model.m.error)
            t = slt_rel.Transition('T_{}'.format(idx), g, r)
            sal_trans_sys.add_transition(t)
        return sal_trans_sys

    @staticmethod
    def sal_module_dmt(dts, vs, pwa_models, init_set, safety_prop, module_name):
        sal_trans_sys = slt_dmt.SALTransSysDMT(dts, module_name, vs, init_set, safety_prop)
        for dt, pwa_model in pwa_models.iteritems():
            # replace decimal point with _ else SAL will throw an
            # error due to incorrect identifier
            dt_str = str(dt).replace('.', '_')
            for idx, sub_model in enumerate(pwa_model):
                g = slt_dmt.Guard(sub_model.p.C, sub_model.p.d)
                r = slt_dmt.Reset(sub_model.m.A, sub_model.m.b)
                t = slt_dmt.Transition(
                        dt, dts, 'C_{}_{}'.format(idx, dt_str), g, r)
                sal_trans_sys.add_transition(t)
        return sal_trans_sys

    @staticmethod
    def sal_module_dft(vs, pwa_model, init_set, safety_prop, module_name):
        sal_trans_sys = slt_dft.SALTransSys(module_name, vs, init_set, safety_prop)

        for idx, sub_model in enumerate(pwa_model):
            g = slt_dft.Guard(sub_model.p.C, sub_model.p.d)
            g.vs = vs
            r = slt_dft.Reset(sub_model.m.A, sub_model.m.b, sub_model.m.error)
            r.vs = vs
            r.vs_ = vs
            t = slt_dft.Transition('C_{}'.format(idx), g, r)
            sal_trans_sys.add_transition(t)
        return sal_trans_sys

    def check(self, depth):
        yices2_not_found = 'yices2: not found'

        self.dump()

        try:
            sal_path_ = os.environ[SAL_PATH] + SAL_INF_BMC
        except KeyError:
            raise err.Fatal("SAL environment variable is not defined. It\n"
                            "should point to sal's top-level directory")
            #raise KeyError

        sal_path = fops.sanitize_path(sal_path_)
        verbosity = 3

        sal_cmd = sal_run_cmd(
                    sal_path,
                    depth,
                    self.module_name,
                    self.prop_name,
                    yices=2,
                    verbosity=verbosity)

        try:
            sal_op = U.strict_call_get_op(sal_cmd)
        except U.CallError as e:
            if yices2_not_found in e.message:
                print 'SAL can not find yices2. Trying with yices...'
                sal_cmd = sal_run_cmd(
                            sal_path,
                            depth,
                            self.module_name,
                            self.prop_name,
                            yices=1,
                            verbosity=verbosity)
                sal_op = U.strict_call_get_op(sal_cmd)
            else:
                raise err.Fatal('unknown SAL error!')

        print sal_op
        self.trace = sal_op_parser.parse_trace(sal_op)
        if self.trace is None:
            print 'BMC failed to find a CE'
            return InvarStatus.Unknown
        else:
            print '#'*40
            print '# Cleaned up trace'
            print '#'*40
            print self.trace
            print '#'*40
            return InvarStatus.Unsafe

    def dump(self):
        sal_file = self.module_name + '.sal'
        fops.write_data(sal_file, str(self.sal_trans_sys))
        return

    def get_last_trace(self):
        """Returns the last trace found or None if no trace exists."""
        return self.trace
