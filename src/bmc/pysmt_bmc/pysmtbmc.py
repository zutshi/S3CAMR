""" Implement the BMC interface that use pysmt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bmc.bmc_spec import BMCSpec, InvarStatus
from bmc.sal_bmc.trace import Trace as BmcTrace
from bmc.sal_bmc.sal_op_parser import Step, Assignment
from bmc.pysmt_bmc.pwa2ts import PWA2TS
from bmc.pysmt_bmc.bmc import BMC as BMCImpl
from bmc.sal_bmc.pwa2salconverter import PWATRACE

class BMC(BMCSpec):
    def __init__(self, vs, pwa_graph, init_cons, final_cons, init_ps, final_ps,
                 fname_constructor, module_name, model_type, smt_engine):
        assert model_type == "dft"

        self.smt_engine = smt_engine
        if model_type != "dft":
            raise Exception('unknown model type')

        self.converter = PWA2TS(module_name, init_cons, final_cons,
                                pwa_graph, vs, init_ps, final_ps)
        self.trace = None
        self.pwa_trace = None

        # List of cex to be removed from the 
        # bmc exploration
        self.cexes_to_remove = []

    def trace_generator(self, depth):
        has_more_disc_trace = True

        while (has_more_disc_trace):
            status = self.check(depth)
            if status == InvarStatus.Unsafe:
                yield self.trace, self.get_pwa_trace()
            else:
                has_more_disc_trace = False
            return

    def check(self, depth):
        ts = self.converter.get_ts()

        # TODO: add cex to avoid

        bmc = BMCImpl(ts.helper, ts, ts.final,self.smt_engine,
                      self.cexes_to_remove)
        res_cex = bmc.find_bug(depth, False)

        if res_cex is None:
            print('BMC failed to find a CE')
            return InvarStatus.Unknown
        else:
            self._build_trace(ts, res_cex)
            return InvarStatus.Unsafe        

    def dump(self):
        raise NotImplementedError

    def get_trace(self):
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
        return self.pwa_trace


    def _build_trace(self, ts, res_cex):
        self.trace = None
        self.pwa_trace = None

        cex_to_remove = []
        all_steps = []
        partitions = []
        models = []

        step_number = -1
        for cex_at_i in res_cex:
            step_number += 1
            inserted = 0
            assignment_list = [step_number]
            trace_list = [None for v in self.converter.vs]
            for var,value in cex_at_i.iteritems():
                # get the index of the variable
                index = self.converter.get_index(var)

                if (index >= 0):
                    float_value = self.converter.to_float(value)
                    src_var = self.converter.vs[index]
                    # Don't know why the 2nd element is ignored in Step...                    
                    assignment_list.append(Assignment([src_var, "", str(float_value)]))
                    inserted += 1
            assert(inserted == len(self.converter.vs))
            # Add a fake assignment... (see sal_op_parser.py:76)
            assignment_list.append({})
            step = Step(assignment_list)
            all_steps.append(step)

            
            # c_val = self.converter._loc_enc.get_counter_value(self.converter._get_loc_var_name(),
            #                                                   cex_at_i,
            #                                                   True)
            # assert c_val in self.converter.val2loc
            # loc = self.converter.val2loc[c_val]
            loc = self.converter.get_loc(cex_at_i)
            # append a new state to the cex_to_remove
            cex_to_remove.append(self.converter._get_loc_enc(loc))

            # Do not process the last state (trans is an input!)
            if (step_number + 1 < len(res_cex)):
                # edge_val = self.converter._loc_enc.get_counter_value(self.converter._get_edge_var_name(),
                #                                                      cex_at_i,
                #                                                      True)
                # if not edge_val in self.converter.val2edge:
                #     print(step_number)
                #     print(self.converter.val2edge)
                # assert edge_val in self.converter.val2edge
                # edge = self.converter.val2edge[edge_val]

                edge = self.converter.get_edge(cex_at_i)

                models.append(self.converter.pwa_graph.edge_m(edge))
                partitions.append(self.converter.pwa_graph.node_p(edge[0]))        

                # Second last state - last transition - add last partition
                if (step_number + 2 == len(res_cex)):
                    partitions.append(self.converter.pwa_graph.node_p(edge[1]))            

        self.pwa_trace = PWATRACE(partitions, models)
        self.trace = BmcTrace(all_steps, self.converter.vs)

        self.cexes_to_remove.append(cex_to_remove)

        # # DEBUG
        # print(self.trace)
        # print("PWA TRACE - partitions")
        # for p in self.pwa_trace.partitions:
        #     print(p.ID)
        # print("PWA TRACE - models")
        # for p in self.pwa_trace.models:
        #     print("%s:" % (str(p) ))

    def gen_new_disc_trace(self):
        raise NotImplementedError
