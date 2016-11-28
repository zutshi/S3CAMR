from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from bmc.bmc_spec import BMCSpec, InvarStatus, PWATRACE
import lin_prog.analyzepath as azp
from modeling.pwa import pwagraph

#import fileops as fops
import utils as U
#import err

#import settings

from IPython import embed

logger = logging.getLogger(__name__)


class BMC(BMCSpec):
    def __init__(self, sys, prop, vs, pwa_model, init_state, safety_prop,
                 init_partitions, prop_partitions, fname_constructor,
                 module_name, model_type):

        assert(isinstance(pwa_model, pwagraph.PWAGraph))
        assert(model_type == 'dft')

        self.trace = None
        self.vs = vs
        sources = [p.ID for p in init_partitions]
        targets = [p.ID for p in prop_partitions]
        #path_gen = pwa_model.get_all_path_generator(sources, targets, 1000000, 10000)
        path_gen = pwa_model.get_all_path_generator(sources, targets)
        for path in path_gen:
            ptrace = [pwa_model.node_p(qi) for qi in path]
            mtrace = [pwa_model.edge_m((qi, qj)) for qi, qj in U.pairwise(path)]
            pwa_trace = PWATRACE(partitions=ptrace, models=mtrace)
            ret_val = azp.overapprox_x0(sys.num_dims, prop, pwa_trace)
#             if ret_val is not None:
#                 embed()
#                 U.pause()
        return

    def check(*args):
        return InvarStatus.Safe

    def get_last_trace(self):
        return None

    def get_new_disc_trace(self):
        return None

    def get_last_pwa_trace(self):
        return None
