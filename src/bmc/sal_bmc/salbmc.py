from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

#from . import saltrans_rel as slt_rel
from . import saltrans_dft as slt_dft
#from . import transitions as trans

#from . import saltrans_dmt as slt_dmt
from bmc.bmc_spec import BMCSpec, InvarStatus, PWATRACE
from . import sal_op_parser
from . import pwa2salconvertor as pwa2sal

import fileops as fops
import utils as U
import err

logger = logging.getLogger(__name__)

SAL_PATH = 'SAL_PATH'
SAL_INF_BMC = '''/bin/sal-inf-bmc'''


class SALBMCError(Exception):
    pass


class SalOpts():
    def __init__(self):
        self.yices = 2
        self.verbosity = 3
        self.iterative = False
        self.preserve_tmp_files = True


# Must separate the arguements. i.e., -v 3 should be given as ['-v', '3']
# This can be avoided by using shell=True, but that is a security risk
def sal_run_cmd(sal_path, depth, sal_file, prop_name, opts=SalOpts()):
    cmd = [
        sal_path,
        '-v', str(opts.verbosity),
        '-d', str(depth),
        #'{}.sal'.format(module_name),
        sal_file,
        prop_name
    ]

    if opts.yices == 2:
        cmd.extend(['-s', 'yices2'])

    if opts.preserve_tmp_files:
        cmd.append('--preserve-tmp-files')

    if opts.iterative:
        cmd.append('-it')

    print(' '.join(cmd))

    return cmd


class BMC(BMCSpec):
    def __init__(self, vs, pwa_model, init_state, safety_prop,
                 prop_partitions, fname_constructor, module_name, model_type):
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

        self.prop_name = 'safety'
        self.fname_constructor = fname_constructor
        self.module_name = module_name
        fname = module_name + '.sal'
        self.sal_file = fname_constructor(fname)
        self.trace = None
        self.vs = vs

        if model_type == 'dft':
            self.sal_trans_sys = self.sal_module_dft(
                    vs, pwa_model, init_state, safety_prop,
                    prop_partitions, module_name)

        elif model_type == 'dmt':
            raise NotImplementedError
            dts = pwa_model.keys()
            self.sal_trans_sys = BMC.sal_module_dmt(
                    dts, vs, pwa_model, init_state, safety_prop, module_name)
        elif model_type == 'ct':
            raise NotImplementedError
        elif model_type == 'rel':
            raise NotImplementedError
        else:
            raise SALBMCError('unknown model type')

        return

    def sal_module_dft(self, vs, pwa_model, init_set, safety_prop,
                       prop_partitions, module_name):

        sal_transitions, self.sal2pwa_map, partid2Cid = pwa2sal.convert_transitions(pwa_model, vs)
        sal_trans_sys = slt_dft.SALTransSys(module_name, vs, init_set, safety_prop, sal_transitions, partid2Cid)

        logger.debug('================ bmc - pwa conversion dict ==================')
        logger.debug(self.sal2pwa_map)
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

        sal_cmd = sal_run_cmd(
                    sal_path,
                    depth,
                    self.sal_file,
                    self.prop_name,
                    )

        try:
            sal_op = U.strict_call_get_op(sal_cmd)
        except U.CallError as e:
            if yices2_not_found in e.message:
                print('SAL can not find yices2. Trying with yices...')
                opts = SalOpts()
                opts.yices = 1
                sal_cmd = sal_run_cmd(
                            sal_path,
                            depth,
                            self.sal_file,
                            self.prop_name,
                            opts)
                sal_op = U.strict_call_get_op(sal_cmd)
            else:
                raise err.Fatal('unknown SAL error!')

        print(sal_op)
        self.trace = sal_op_parser.parse_trace(sal_op, self.vs)
        if self.trace is None:
            print('BMC failed to find a CE')
            return InvarStatus.Unknown
        else:
            #self.trace.set_vars(self.vs)
            print('#'*40)
            print('# Cleaned up trace')
            print('#'*40)
            print(self.trace)
            print('#'*40)
            return InvarStatus.Unsafe

    def dump(self):
        fops.write_data(self.sal_file, str(self.sal_trans_sys))
        return

    def get_last_trace(self):
        """Returns the last trace found or None if no trace exists."""
        return self.trace

    def get_last_pwa_trace(self):
        """Converts a bmc trace to a sequence of sub_models in the original pwa.

        Parameters
        ----------

        Returns
        -------
        pwa_trace = [sub_model_0, sub_model_1, ... ,sub_model_n]
        pwa_trace =
            models = [m01, m12, ... , m(n-1)n]
            partitions = [p0, p1, p2, ..., pn]

        Notes
        ------
        For now, pwa_trace is only a list of sub_models, as relational
        modeling is being done with KMIN = 1. Hence, there is no
        ambiguity.
        """

        steps = list(self.trace)
#         # each step, but the last, corresponds to a transition
#         for step in steps[:-1]:
#             part_id = self.sal2pwa_map[step.assignments['cell']]
#             sub_model = self.sal2pwa_map[step.tid]

#             # Assumption of trace building is that each submodel only
#             # has 1 unique next location. If this violated, we need to
#             # add cell ids/part ids to resolve the ambiguity.
#             assert(len(sub_model.pnexts) == 1)

#             assert(sub_model.p.ID == part_id)
#             # this is still untested, so in case assert is off...
#             assert(sub_model.p.ID == part_id)
#                 #err.warn('gone case')

#             #pwa_trace.extend((part_id, sub_model))
#             pwa_trace.append(sub_model)

        s2p = self.sal2pwa_map
        models = [s2p[step.tid].m for step in steps[:-1]]
        #partitions = [self.sal2pwa_map[step.assignments['cell']] for step in steps]
        partitions = [s2p[step.tid].p for step in steps[:-1]]
        partitions.append(s2p[step.tid].pnext)

        # append the last location/cell/partition id
        #last_step = steps[-1]
        #last_part_id = self.sal2pwa_map[last_step.assignments['cell']]
        #pwa_trace.append(last_part_id)
        pwa_trace = PWATRACE(partitions, models)
        return pwa_trace

    def get_new_disc_trace(self):
        """makes trace = None, signifying no more traces..."""
        self.trace = None
        return


################################################
# ############# CEMETERY #######################
################################################

#     @staticmethod
#     def sal_module_dmt(dts, vs, pwa_models, init_set, safety_prop, module_name):
#         sal_trans_sys = slt_dmt.SALTransSysDMT(dts, module_name, vs, init_set, safety_prop)
#         for dt, pwa_model in pwa_models.iteritems():
#             # replace decimal point with _ else SAL will throw an
#             # error due to incorrect identifier
#             dt_str = str(dt).replace('.', '_')
#             for idx, sub_model in enumerate(pwa_model):
#                 g = slt_dmt.Guard(sub_model.p.C, sub_model.p.d)
#                 r = slt_dmt.Reset(sub_model.m.A, sub_model.m.b)
#                 t = slt_dmt.Transition(
#                         dt, dts, 'C_{}_{}'.format(idx, dt_str), g, r)
#                 sal_trans_sys.add_transition(t)
#         return sal_trans_sys
