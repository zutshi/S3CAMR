"""Generic interfaces to perform BMC.

It defines the results (e.g. traces, status).

"""

import abc


class InvarStatus:
    Safe, Unsafe, Unknown = range(3)


class BMCSpec():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check(self, depth):
        """Returns one value from InvarStatus"""
        return

    @abc.abstractmethod
    def get_last_trace(self):
        """Returns the last trace found or None if no trace exists."""
        return
