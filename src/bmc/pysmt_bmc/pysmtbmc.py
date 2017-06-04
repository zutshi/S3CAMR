""" Implement the BMC interface that use pysmt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bmc.bmc_spec import BMCSpec, InvarStatus
from bmc.pysmt_bmc.pwa2ts import PWA2TS
from bmc.pysmt_bmc.bmc import BMC

class BMC(BMCSpec):
    def __init__(self, vs, pwa_graph, init_cons, final_cons, init_ps, final_ps,
                 fname_constructor, module_name, model_type):
        assert model_type == "dft"

        if model_type != "dft":
            raise Exception('unknown model type')

        self.converter = PWA2TS(module_name, init_cons, final_cons,
                                pwa_graph, vs, init_ps, final_ps)

    def trace_generator(self, depth):
        raise NotImplementedError

        for i in range(1):
            status = self.check(depth)
            if status == InvarStatus.Unsafe:
                yield self.trace, self.get_pwa_trace()
            return



    def check(self, depth):
        ts = self.converter.get_ts()

        bmc = BMC(ts.helper, ts, ts.final)    

        # TODO: store the trace
        res_cex = bmc.find_bug(depth, False)

        if res_cex is None:
            print('BMC failed to find a CE')
            return InvarStatus.Unknown
        else:
            # TODO Extract trace
            self.trace = None
            return InvarStatus.Unsafe

    def dump(self):
        raise NotImplementedError

    def get_trace(self):
        raise NotImplementedError
        """Returns the last trace found or None if no trace exists."""
        return self.trace

    def get_last_traces(self):
        raise NotImplementedError
        # return None, None

    def get_pwa_trace(self):
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
        raise NotImplementedError


    def gen_new_disc_trace(self):
        raise NotImplementedError
