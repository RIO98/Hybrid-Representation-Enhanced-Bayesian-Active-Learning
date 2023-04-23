from __future__ import absolute_import

from abc import abstractmethod

from ..augmentor.base_operation import Operation


class Operation(Operation):
    """ Base class of operations."""

    def preprocess(self, x):
        x = self.preprocess_core(x)
        return x

    def postprocess(self, x):
        x = self.postprocess_core(x)
        return x

    @abstractmethod
    def apply_core(self, x):
        raise NotImplementedError()

    def apply(self, x):
        x = self.preprocess(x)
        x = self.apply_core(x)
        x = self.postprocess(x)
        return x

