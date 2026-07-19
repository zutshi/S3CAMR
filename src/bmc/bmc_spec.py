"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""


import abc

import numpy as np


class InvarStatus:
    Safe, Unsafe, Unknown = range(3)


class BMCSpec(metaclass=abc.ABCMeta):
    """BMCSpec
    Defines the spec for a BMC engine"""
    #__metaclass__ = abc.ABCMeta

    # supersedes previous ways
    @abc.abstractmethod
    def trace_generator(self):
        """Returns trace generator"""
        raise NotImplementedError

    @abc.abstractmethod
    def check(self, depth):
        """Returns one value from InvarStatus"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_trace(self):
        """Returns the last trace found or None if no trace exists."""
        raise NotImplementedError

    @abc.abstractmethod
    def gen_new_disc_trace(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_pwa_trace(self):
        raise NotImplementedError


class TraceSimple(metaclass=abc.ABCMeta):
    """Simple Trace: provides minimal functionality"""

    @abc.abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_array(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError
