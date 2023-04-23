import json
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np

from .base_operation import Operation

_channel_axis = 0


class DataAugmentor(object):
    """
    Data augmentor for image and volume data.
    This class manages the operations.

    :param n_dim: Number of dimensions of the input data, optional, default is None.
    :type n_dim: Optional[int]
    """

    def __init__(self, n_dim: Optional[int] = None) -> None:
        assert n_dim is None or n_dim in [2, 3]

        self._n_dim = n_dim
        self._operations = []

    @staticmethod
    def postprocess(x: Optional[np.ndarray], is_expanded: bool) -> Optional[np.ndarray]:
        """
        Postprocess the input data after applying operations.

        :param x: Input data to postprocess.
        :type x: Optional[np.ndarray]
        :param is_expanded: Boolean indicating if the data was expanded.
        :type is_expanded: bool

        :return: Postprocessed input data.
        :rtype: Optional[np.ndarray]
        """
        if not is_expanded:
            return x

        if x is not None:
            if isinstance(x, list):
                x = [np.rollaxis(x_i, _channel_axis, 0)[0] for x_i in x]
            else:
                x = np.rollaxis(x, _channel_axis, 0)[0]
        return x

    def add(self, op: Operation) -> None:
        """
        Add an operation to the data augmentor.

        :param op: The operation to add.
        :type op: Operation
        """

        assert isinstance(op, Operation)
        if self._n_dim is None:  # NOTE: auto set
            self._n_dim = op.ndim
        self._operations.append(op)

    def get(self) -> List[Operation]:
        """
        Get the list of operations in the data augmentor.

        :return: List of operations.
        :rtype: List[Operation]
        """

        return self._operations

    def preprocess(self, x: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
        """
        Preprocess the input data before applying operations.

        :param x: Input data to preprocess.
        :type x: Optional[np.ndarray]

        :return: Preprocessed input data and a boolean indicating if the data was expanded.
        :rtype: Tuple[Optional[np.ndarray], bool]
        """

        is_expanded = False

        if x is not None:
            if isinstance(x, list):
                if x[0].ndim == self._n_dim:
                    x = [np.expand_dims(x_i, _channel_axis) for x_i in x]
                    is_expanded = True
                assert x[0].ndim == self._n_dim + 1, '`x[0].ndim` must be `self._n_dim + 1`'
            else:
                if x.ndim == self._n_dim:
                    x = np.expand_dims(x, _channel_axis)
                    is_expanded = True
                assert x.ndim == self._n_dim + 1, '`x.ndim` must be `self._n_dim + 1`'

        return x, is_expanded

    def apply(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply the operations on the input data.

        :param x: Input data to apply operations, optional, default is None.
        :type x: Optional[np.ndarray]
        :param y: label data to apply operations, optional, default is None.
        :type y: Optional[np.ndarray]

        :return: The processed input data.
        :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        """

        x, is_expanded_x = self.preprocess(x)
        y, is_expanded_y = self.preprocess(y)

        for op in self._operations:
            x, y = op.apply(x, y)

        x = self.postprocess(x, is_expanded_x)
        y = self.postprocess(y, is_expanded_y)

        assert (x is not None or y is not None)
        if x is None:
            return y
        if y is None:
            return x
        return x, y

    def __call__(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        return self.apply(x, y)

    def summary(self, out: Optional[str] = None) -> OrderedDict:
        """
        Generate a summary of the operations in the data augmentor.

        :param out: Optional output file path to save the summary, default is None.
        :type out: Optional[str]

        :return: An ordered dictionary containing the summary of the operations.
        :rtype: OrderedDict
        """

        ret = OrderedDict()

        for op in self._operations:
            name = op.__class__.__name__

            # Prevent name conflict in the dictionary.
            if name in ret.keys():
                cnt = 1
                while True:
                    _name = '%s_%d' % (name, cnt)
                    if _name not in ret.keys():
                        break
                    cnt += 1
                name = _name

            args = op.summary().copy()
            ignore_keys = ['__class__', 'self']
            for key in ignore_keys:
                if key in args.keys():
                    args.pop(key)

            ret[name] = args

        if out is None:
            return ret

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(ret, f, ensure_ascii=False, indent=4)

        return ret


if __name__ == '__main__':
    operator = DataAugmentor()
    print(operator.summary())
