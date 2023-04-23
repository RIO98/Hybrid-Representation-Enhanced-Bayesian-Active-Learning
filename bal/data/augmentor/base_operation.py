from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union, Tuple, Optional

import numpy as np


class Operation(metaclass=ABCMeta):
    """ Base class of operations."""

    def __init__(self, *args, **kwargs):
        self._args = locals()

    @staticmethod
    def preprocess_core(x: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Core preprocessing function.

        :param x: The input data.
        :type x: numpy.ndarray or list of numpy.ndarray
        :return: The preprocessed input data.
        :rtype: list of numpy.ndarray
        """
        if x is None:
            return x
        elif isinstance(x, list):
            return x
        else:
            return [x]  # NOTE: to list

    def preprocess(self, x: Union[np.ndarray, List[np.ndarray]], y: Optional[Union[np.ndarray, List[np.ndarray]]]) -> \
            Tuple[Union[np.ndarray, List[np.ndarray]], Optional[Union[np.ndarray, List[np.ndarray]]]]:

        """
        Preprocesses the input data.

        :param x: The input data to be augmented.
        :type x: numpy.ndarray or list of numpy.ndarray
        :param y: The label data to be augmented.
        :type y: numpy.ndarray or list of numpy.ndarray or None
        :return: A tuple containing the preprocessed input and label data.
        :rtype: tuple of numpy.ndarray or list of numpy.ndarray
        """
        x = self.preprocess_core(x)
        y = self.preprocess_core(y)
        return x, y

    @staticmethod
    def postprocess_core(x: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, None]:
        """
        Core postprocessing function.

        :param x: The input data.
        :type x: numpy.ndarray or list of numpy.ndarray
        :return: The postprocessed input data.
        :rtype: numpy.ndarray or None
        """
        if x is None:
            return x
        elif len(x) == 1:
            return x[0]
        else:
            return x

    def postprocess(self, x: Optional[Union[np.ndarray, List[np.ndarray]]],
                    y: Optional[Union[np.ndarray, List[np.ndarray]]]) -> Tuple[
        Optional[Union[np.ndarray, List[np.ndarray]]], Optional[Union[np.ndarray, List[np.ndarray]]]]:
        """
        Postprocesses the output data.

        :param x: The output data.
        :type x: numpy.ndarray or list of numpy.ndarray or None
        :param y: The label data.
        :type y: numpy.ndarray or list of numpy.ndarray or None
        :return: A tuple containing the postprocessed output and label data.
        :rtype: tuple of numpy.ndarray or list of numpy.ndarray or None
        """
        x = self.postprocess_core(x)
        y = self.postprocess_core(y)
        return x, y

    @abstractmethod
    def apply_core(self, x, y):
        raise NotImplementedError()

    def apply(self, x=None, y=None):
        """
        Applies the augmentations to the input data.

        :param x: The input data to be augmented.
        :type x: numpy.ndarray or list of numpy.ndarray or None
        :param y: The label data to be augmented.
        :type y: numpy.ndarray or list of numpy.ndarray or None
        :return: A tuple containing the augmented input and label data.
        :rtype: tuple of numpy.ndarray or list of numpy.ndarray or None
        """
        x, y = self.preprocess(x, y)
        x, y = self.apply_core(x, y)
        x, y = self.postprocess(x, y)
        return x, y

    @property
    @abstractmethod
    def ndim(self):
        raise NotImplementedError()

    def summary(self) -> Dict:
        """
        Prints a summary of the operation.

        :return: A dictionary containing the summary of the operation.
        :rtype: dict
        """
        return self._args
