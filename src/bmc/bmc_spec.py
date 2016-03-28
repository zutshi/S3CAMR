import abc


class BMCSpec():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check(self, depth):
        return
