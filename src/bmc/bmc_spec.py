"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc

import numpy as np
import six


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
    def get_last_bmc_trace(self):
        """Returns the last trace found or None if no trace exists."""
        return

    @abc.abstractmethod
    def get_new_disc_trace(self):
        return

    @abc.abstractmethod
    def get_last_pwa_trace(self):
        return

    @abc.abstractmethod
    def get_last_traces(self):
        return

@six.add_metaclass(abc.ABCMeta)
class TraceSimple(object):
    """Simple Trace: provides minimal functionality"""

    @abc.abstractmethod
    def __getitem__(self, idx):
        return

    @abc.abstractmethod
    def __iter__(self):
        return

    @abc.abstractmethod
    def to_array(self):
        return

    @abc.abstractmethod
    def __len__(self):
        return

    @abc.abstractmethod
    def __str__(self):
        return
