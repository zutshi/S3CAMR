"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""

import abc
import itertools as it
import numpy as np

class InvarStatus:
    Safe, Unsafe, Unknown = range(3)


class BMCSpec():
    """BMCSpec
    Defines the spec for a BMC engine"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check(self, depth):
        """Returns one value from InvarStatus"""
        return

    @abc.abstractmethod
    def get_last_trace(self):
        """Returns the last trace found or None if no trace exists."""
        return

    @abc.abstractmethod
    def get_new_disc_trace(self):
        return

    @abc.abstractmethod
    def get_last_pwa_trace(self):
        return


class TraceSimple(object):
    """Simple Trace: provides minimal functionality"""
    def __init__(self, trace):
        self.xvars = None
        self.trace = trace
#         for step in trace:
#             for ass in step.assignments:
#                 print ass.lhs, ass.rhs
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

    def set_vars(self, vs):
        self.vs = vs
        return

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
