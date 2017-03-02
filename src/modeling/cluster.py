from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import six
import abc

import numpy as np


import constraints as cons
import utils as U

import globalopts

logger = logging.getLogger(__name__)


def factory():
    if globalopts.opts.clustering == 'cell':
        return cell
    elif globalopts.opts.clustering == 'box':
        return box
    elif globalopts.opts.clustering == 'hull':
        return hull
    else:
        raise NotImplementedError('unknown clustering method requested')


@six.add_metaclass(abc.ABCMeta)
class ClusterModel(object):
    #__metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, x):
        return

    @abc.abstractmethod
    def poly(self):
        """ return C, d """
        return


# class Box(ClusterModel):
#     def __init__(self, x):
#         self.ival_cons = cons.IntervalCons(np.min(x, axis=0),
#                                            np.max(x, axis=0))

#     def poly(self):
#         return self.ival_cons.poly()

def box(_, x):
    ival_cons = cons.IntervalCons(np.min(x, axis=0), np.max(x, axis=0))
    return ival_cons.poly()


# class Q(ClusterModel):
#     def __init__(self, qi):
#         self.qi = qi

#     def poly(self):
#         return self.qi.ival_constraints.poly()

def cell(qi, _):
    return qi.ival_constraints.poly()


# class Hull(ClusterModel):
#     def __init__(self, x):
#         self.x = x

#     # TODO: memoize
#     def poly(self):
#         C, d = U.poly_v2h(self.x)
#         return C, d

def hull(_, x):
    C, d = U.poly_v2h(x)
    return C, d
