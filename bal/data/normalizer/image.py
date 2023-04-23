# Original code by Yuta Hiasa (2020) https://github.com/yuta-hi/pytorch_bayesian_unet
# Modified by Ganping Li (2023) https://github.com/RIO98/Hybrid-Representation-Enhanced-Bayesian-Active-Learning

from __future__ import absolute_import

from typing import List, Union, Tuple

import numpy as np

from .base_operation import Operation

_row_axis = 1
_col_axis = 2
_channel_axis = 0


class Quantize(Operation):
    """
    Quantize the given images to specific resolution.

    Non-linearity and overfitting make the neural networks sensitive to tiny noises in high-dimensional data [1].
    Quantizing it to a necessary and sufficient level may be effective, especially for medical images which have >16 bits information.
    [1] Goodfellow et al., "Explaining and harnessing adversarial examples.", 2014. https://arxiv.org/abs/1412.6572

    :param n_bit: Number of bits.
    :type n_bit: int
    :param x_min: Minimum value in the input domain, defaults to 0.
    :type x_min: float, optional
    :param x_max: Maximum value in the input domain, defaults to 1.
    :type x_max: float, optional
    :param rescale: If True, output value is rescaled to input domain, defaults to True.
    :type rescale: bool, optional
    """

    def __init__(self, n_bit, x_min=0., x_max=1., rescale=True):
        self._args = locals()
        self._args.pop('self')
        self._n_bit = n_bit
        self._x_min = x_min
        self._x_max = x_max
        self._rescale = rescale
        self._ndim = 2

    @staticmethod
    def quantize(x: np.ndarray, n_bit: int, x_min: float = 0., x_max: float = 1., rescale: bool = True) -> np.ndarray:
        """
        Quantize the input values.

        :param x: Input values.
        :type x: ndarray
        :param n_bit: Number of bits.
        :type n_bit: int
        :param x_min: Minimum value in the input domain, defaults to 0.
        :type x_min: float, optional
        :param x_max: Maximum value in the input domain, defaults to 1.
        :type x_max: float, optional
        :param rescale: If True, output value is rescaled to input domain, defaults to True.
        :type rescale: bool, optional
        :return: Quantized values.
        :rtype: ndarray
        """

        n_discrete_values = 1 << n_bit
        scale = (n_discrete_values - 1) / (x_max - x_min)
        quantized = np.round(x * scale) - np.round(x_min * scale)
        quantized = np.clip(quantized, 0., n_discrete_values)

        if not rescale:
            return quantized

        quantized /= scale
        return quantized + x_min

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x: List[np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        x = [self.quantize(x_i, self._n_bit, self._x_min, self._x_max, self._rescale) for x_i in x]
        return x


class Clip(Operation):
    """
    Clip (limit) the values in given images.

    :param param: Tuple of minimum and maximum values.
        If 'minmax' or 'ch_minmax', minimum and maximum values are automatically estimated.
        'ch_minmax' is the channel-wise minmax normalization.
    :type param: tuple or str
    """

    def __init__(self, param: Union[Tuple[float, float], str]) -> None:
        self._args = locals()
        self._args.pop('self')
        self._param = param
        self._ndim = 2

    @staticmethod
    def clip(x: np.ndarray, param: Union[Tuple[float, float], str], scale: float = 1.) -> np.ndarray:
        """
        Clip the input values.

        :param x: Input values.
        :type x: np.ndarray
        :param param: Tuple of minimum and maximum values.
            If 'minmax' or 'ch_minmax', minimum and maximum values are automatically estimated.
            'ch_minmax' is the channel-wise minmax normalization.
        :type param: tuple or str
        :param scale: Scale factor, defaults to 1.
        :type scale: float, optional
        :return: Clipped values.
        :rtype: np.ndarray
        """

        if isinstance(param, str):
            if param == 'minmax':
                param = (np.min(x), np.max(x))
            elif param == 'ch_minmax':
                tmp = np.swapaxes(x, _channel_axis, 0)
                tmp = np.reshape(tmp, (len(tmp), -1))
                tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
                param = (np.min(tmp, axis=1).reshape(tmp_shape),
                         np.max(tmp, axis=1).reshape(tmp_shape))
            else:
                raise NotImplementedError('unsupported parameters..')

        assert isinstance(param, (list, tuple))
        x = np.clip(x, param[0], param[1])
        x = (x - param[0]) / (param[1] - param[0])  # [0, 1]

        return x * scale

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x: List[np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        x = [self.clip(x_i, self._param) for x_i in x]
        return x


class Subtract(Operation):
    """
    Subtract a value or tensor from given images.

    :param param: A value or tensor.
        If 'mean' or 'ch_mean', subtracting values are automatically estimated.
        'ch_mean' is to subtract the channel-wise mean.
    :type param: float, numpy.ndarray or str
    """

    def __init__(self, param: Union[float, np.ndarray, str]) -> None:
        self._args = locals()
        self._param = param
        self._ndim = 2

    @staticmethod
    def subtract(x: np.ndarray, param: Union[float, np.ndarray, str]) -> np.ndarray:
        """
        Subtract the input values.

        :param x: Input values.
        :type x: np.ndarray
        :param param: A value or tensor.
            If 'mean' or 'ch_mean', subtracting values are automatically estimated.
            'ch_mean' is to subtract the channel-wise mean.
        :type param: float, numpy.ndarray or str
        :return: Subtracted values.
        :rtype: np.ndarray
        """

        if isinstance(param, str):
            if param == 'mean':  # NOTE: for z-score normalization
                param = np.mean(x)
            elif param == 'ch_mean':
                tmp = np.swapaxes(x, _channel_axis, 0)
                tmp = np.reshape(tmp, (len(tmp), -1))
                tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
                param = np.mean(tmp, axis=1).reshape(tmp_shape)
            else:
                raise NotImplementedError('unsupported parameters..')

        x -= param

        return x

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x: List[np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        x = [self.subtract(x_i, self._param) for x_i in x]
        return x


class Divide(Operation):
    """
    Divide the given images by a value or tensor.

    :param param: A value or tensor.
        If 'std' or 'ch_std', dividing values are automatically estimated.
        'ch_std' is to divide the channel-wise standard deviation.
    :type param: float, numpy.ndarray or str
    """

    def __init__(self, param: Union[float, np.ndarray, str]) -> None:
        self._args = locals()
        self._param = param
        self._ndim = 2

    @staticmethod
    def divide(x: np.ndarray, param: Union[float, np.ndarray, str]) -> np.ndarray:
        """
        Divide the input values.

        :param x: Input values.
        :type x: np.ndarray
        :param param: A value or tensor.
            If 'std' or 'ch_std', dividing values are automatically estimated.
            'ch_std' is to divide the channel-wise standard deviation.
        :type param: float, numpy.ndarray or str
        :return: Divided values.
        :rtype: np.ndarray
        """

        if isinstance(param, str):
            if param == 'std':  # NOTE: for z-score normalization
                param = np.std(x)
            elif param == 'ch_std':
                tmp = np.swapaxes(x, _channel_axis, 0)
                tmp = np.reshape(tmp, (len(tmp), -1))
                tmp_shape = [len(tmp)] + [1] * (x.ndim - 1)
                param = np.std(tmp, axis=1).reshape(tmp_shape)
            else:
                raise NotImplementedError('unsupported parameters..')

        x /= param

        return x

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x: List[np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        x = [self.divide(x_i, self._param) for x_i in x]
        return x
