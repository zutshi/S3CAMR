"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""

import abc


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


class TraceSimple(object):
    """Simple Trace: provides minimal functionality"""
    def __init__(self, trace):
        self.trace = trace
#         for step in trace:
#             for ass in step.assignments:
#                 print ass.lhs, ass.rhs
        return

    def __getitem__(self, idx):
        return self.trace[idx].assignments

    def to_assignments(self):
        assignments = [
                {ass.lhs: ass.rhs for ass in step.assignments}
                for step in self.trace
                ]
        return assignments

    def __str__(self):
        return '\n'.join(str(step) for step in self.trace)
