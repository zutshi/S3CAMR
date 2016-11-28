"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections

import numpy as np
import six

import settings

PWATRACE = collections.namedtuple('pwatrace', 'partitions models')

class InvarStatus:
    Safe, Unsafe, Unknown = range(3)


@six.add_metaclass(abc.ABCMeta)
class BMCSpec():
    """BMCSpec
    Defines the spec for a BMC engine"""
    #__metaclass__ = abc.ABCMeta

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
    def __init__(self, trace, vs):
        self.xvars = None
        self.trace = trace
        self.vs = vs

        #############################################################
        assert(settings.CE)
        #############################################################
        # Remove the last step to take care of the special case of CE
        # This removes the redundant step, which causes issues later..

        # Make sure last step is CE
        assert(self.trace[-1].assignments['cell'] == 'CE')

        # Make sure that the last two steps are identical
        A1 = self.trace[-1].assignments
        A2 = self.trace[-2].assignments
        for k, d in A1.iteritems():
            assert(type(k) is str)
            if k[0] == 'x' or k[0] == 'w':
                assert(A2[k] == d)

        # Make the 2nd last step look like the last one
        A2['unsafe'] == 'true'
        self.trace[-2].tid = ''
        self.trace = self.trace[0:-1]

        #############################################################

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
