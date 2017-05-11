import bmc.bmc_spec as spec
import numpy as np


class Trace(spec.TraceSimple):
    """Simple Trace: provides minimal functionality"""
    def __init__(self, trace, vs):
        self.xvars = None
        self.trace = trace
        self.vs = vs

#         for step in trace:
#             for ass in step.assignments:
#                 print(ass.lhs, ass.rhs)
        return

    def __getitem__(self, idx):
        return self.trace[idx]

#     def to_assignments(self):
#         assignments = [
#                 {ass.lhs: ass.rhs for ass in step.assignments}
#                 for step in self.trace
#                 ]
#         return assignments
    def __iter__(self):
        return (step for step in self.trace)

    #def set_vars(self, vs):
        #self.vs = vs
        #return

    def to_array(self):
        # vars must have been set before this is called
        assert(self.vs is not None)
        xvars = self.vs

        x_array = []
        for step in self.trace:
            # jth step
            xj = []
            for xi in xvars:
                xival = step.assignments[xi]
                xj.append(xival)
            x_array.append(xj)
        return np.array(x_array)

    def __len__(self):
        return len(self.trace)

    def __str__(self):
        return '\n'.join(str(step) for step in self.trace)
