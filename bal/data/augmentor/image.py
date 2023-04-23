# Original code by Yuta Hiasa (2020) https://github.com/yuta-hi/pytorch_bayesian_unet
# Modified by Ganping Li (2023) https://github.com/RIO98/Hybrid-Representation-Enhanced-Bayesian-Active-Learning

from __future__ import annotations

import copy
from typing import List, Tuple, Union, Any

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from numpy import ndarray

from .base_operation import Operation

_row_axis = 1
_col_axis = 2
_channel_axis = 0


class Flip(Operation):
    """
    Reverse the order of elements in given images along the specified axis, stochastically.

    :param probability: Controls the probability that the operation is performed when it is invoked in the pipeline.
    :type probability: float
    :param axis: An integer axis
    :type axis: int
    """

    def __init__(self, probability: float, axis: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = locals()
        self._args.pop("self")
        self._axis = axis
        self._probability = probability
        self._ndim = 2

    @staticmethod
    def flip(x: np.ndarray, axis: int) -> np.ndarray | None:
        if x is None:
            return x
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply flip operation on input data along the specified axis.

        :param x: Input data
        :param y: Labels
        :return: Tuple of flipped input data and labels
        :rtype: tuple
        """

        if np.random.random() < self._probability:
            if x is not None:
                x = [self.flip(x_i, self._axis) for x_i in x]
            if y is not None:
                y = [self.flip(y_i, self._axis) for y_i in y]
        return x, y


class Crop(Operation):
    """
    Crop given images to the specified size at a random location.

    :param probability: Controls the probability that the operation is performed when it is invoked in the pipeline.
    :type probability: float
    :param size: Cropping size
    :type size: list or tuple
    """

    def __init__(self, probability: float, size: Union[list, tuple], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = locals()
        self._args.pop("self")
        assert (isinstance(size, (list, tuple)))
        self._size = size
        self._probability = probability
        self._ndim = 2

    @staticmethod
    def crop(x: np.ndarray, x_s: int, x_e: int, y_s: int, y_e: int) -> np.ndarray | None:
        if x is None:
            return x
        x = np.asarray(x).swapaxes(_channel_axis, 0)
        x = x[:, x_s:x_e, y_s:y_e]
        x = x.swapaxes(0, _channel_axis)
        return x

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x, y) -> tuple[None, Any] | tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
        """
        Apply crop operation on input data along the specified dimensions.

        :param x: Input data
        :param y: Labels
        :return: Tuple of cropped input data and labels
        :rtype: tuple
        """

        if x is not None:
            h, w = x[0].shape[_row_axis], x[0].shape[_col_axis]
        elif y is not None:
            h, w = y[0].shape[_row_axis], y[0].shape[_col_axis]
        else:
            return x, y

        x_s = np.random.randint(0, h - self._size[0] + 1)
        x_e = x_s + self._size[0]
        y_s = np.random.randint(0, w - self._size[1] + 1)
        y_e = y_s + self._size[1]

        if np.random.random() < self._probability:
            if x is not None:
                x = [self.crop(x_i, x_s, x_e, y_s, y_e) for x_i in x]
            if y is not None:
                y = [self.crop(y_i, x_s, x_e, y_s, y_e) for y_i in y]
        return x, y


class ResizeCrop(Crop):
    """
    Resize given images to a random size, and crop them to the specified size at a random location.

    :param probability: Controls the probability that the operation is performed when it is invoked in the pipeline.
    :type probability: float
    :param resize_size: Resizing size
    :type resize_size: list or tuple
    :param crop_size: Cropping size, which should be smaller than resizing size
    :type crop_size: list or tuple
    """

    def __init__(self,
                 probability: float,
                 resize_size: Union[list, tuple],
                 crop_size: Union[list, tuple],
                 interp_order: Union[int, Tuple[int, int]] = (0, 0)
                 ):
        super(ResizeCrop, self).__init__(probability=1., size=crop_size)

        self._args = locals()
        self._args.pop("self")
        self._probability = probability

        assert (isinstance(resize_size, (list, tuple)))
        self._resize_size = resize_size

        assert all([src >= dst for src, dst in zip(self._resize_size, self._size)]), \
            'Cropping size should be smaller than resizing size..'

        if isinstance(interp_order, int):
            interp_order = [interp_order] * 2
        self._interp_order = interp_order

    @staticmethod
    def resize(x, size: Tuple[int, int], interp_order: int = 0) -> Union[None, np.ndarray]:
        """
        Resize the input data to the specified size using the specified interpolation order.

        :param x: Input data
        :param size: Target size for resizing
        :param interp_order: Interpolation order, 0 for nearest, 1 for linear, and any other value for cubic
        :return: Resized input data or None if input is None
        :rtype: None or np.ndarray
        """

        if x is None:
            return x

        if interp_order == 0:
            interpolation = cv2.INTER_NEAREST
        elif interp_order == 1:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_CUBIC

        x = np.asarray(x).swapaxes(_channel_axis, 2)  # NOTE: opencv's format
        x = cv2.resize(x, size, interpolation=interpolation)
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        x = x.swapaxes(2, _channel_axis)

        return x

    def apply_core(self, x, y) -> tuple[None, Any] | tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
        """
        Apply resize and crop operations on input data and labels.

        :param x: Input data
        :param y: Labels
        :return: Tuple of resized and cropped input data and labels
        :rtype: tuple
        """

        size = (np.random.randint(self._size[0], self._resize_size[0] + 1),
                np.random.randint(self._size[1], self._resize_size[1] + 1))

        if np.random.random() < self._probability:
            if x is not None:
                x = [self.resize(x_i, size, self._interp_order[0]) for x_i in x]
            if y is not None:
                y = [self.resize(y_i, size, self._interp_order[1]) for y_i in y]

        return super().apply_core(x, y)


class RandomErasing(Operation):
    """
    Perform random erasing augmentation on input images.

    :param probability: Controls the probability that the operation is performed when it is invoked in the pipeline.
    :type probability: float
    :param size: Tuple specifying the range of erasing size as a fraction of image size
    :type size: tuple, default (0.05, 0.2)
    :param ratio: Tuple specifying the range of aspect ratio of erasing region
    :type ratio: tuple, default (0.3, 1.0)
    :param value_range: Tuple specifying the range of pixel values for the erasing region
    :type value_range: tuple, default (-150, 350)
    """

    def __init__(self,
                 probability: float,
                 size: Tuple[float, float] = (0.05, 0.2),
                 ratio: Tuple[float, float] = (0.3, 1.0),
                 value_range: Tuple[int, int] = (-150, 350),
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._args = locals()
        self._args.pop("self")

        self._probability = probability
        self._size = size
        self._ratio = ratio
        self._value_range = value_range
        self._ndim = 2

    @staticmethod
    def random_erasing(img: np.ndarray,
                       size: Tuple[float, float],
                       ratio: Tuple[float, float],
                       value_range: Tuple[int, int]) -> np.ndarray:
        """
        Apply random erasing on the input image.

        :param img: Input image
        :param size: Tuple specifying the range of erasing size as a fraction of image size
        :param ratio: Tuple specifying the range of aspect ratio of erasing region
        :param value_range: Tuple specifying the range of pixel values for the erasing region
        :return: Augmented image with random erasing applied
        :rtype: np.ndarray
        """
        x = copy.deepcopy(img)
        c, h, w = x.shape

        mask_area = np.random.randint(h * w * size[0], h * w * size[1])
        mask_aspect_ratio = ratio[0] + np.random.rand() * ratio[1]

        mask_h = int(np.sqrt(mask_area / mask_aspect_ratio))
        mask_w = int(mask_aspect_ratio * mask_h)
        mask_h = min(mask_h, h - 1)
        mask_w = min(mask_w, w - 1)
        # random_value = np.random.uniform(value_range[0], value_range[1], (mask_h, mask_w, c))
        random_value = np.random.uniform(value_range[0], value_range[1], (c, mask_h, mask_w))

        top = np.random.randint(0, h - mask_h)
        left = np.random.randint(0, w - mask_w)

        # x[top:top+mask_h, left:left+mask_w, :] = random_value
        x[:, top:top + mask_h, left:left + mask_w] = random_value
        return x

    def apply_core(self, x: Union[None, np.ndarray], y: Union[None, np.ndarray]) -> \
            tuple[list[ndarray] | None, ndarray | None]:
        """
        Apply random erasing on input data if the random probability is greater than 0.3.

        :param x: Input data
        :param y: Labels
        :return: Tuple of input data and labels after random erasing
        :rtype: tuple
        """

        if x is not None and np.random.random() < self._probability:
            x = [self.random_erasing(x_i, self._size, self._ratio, self._value_range) for x_i in x]
        return x, y

    @property
    def ndim(self) -> int:
        return self._ndim

    def summary(self) -> dict:
        return self._args


class Affine(Operation):
    """
    Apply a randomly generated affine transform matrix to given images.

    :param probability: Controls the probability that the operation is performed when it is invoked in the pipeline.
    :type probability: float
    :param rotation: Rotation angle, defaults to 0.
    :type rotation: float, optional
    :param translate: Translation ratios, defaults to (0., 0.).
    :type translate: Tuple[float, float], optional
    :param shear: Shear angle, defaults to 0.
    :type shear: float, optional
    :param zoom: Enlarge and shrinkage ratios, defaults to (1., 1.).
    :type zoom: Tuple[float, float], optional
    :param keep_aspect_ratio: Keep the aspect ratio, defaults to True.
    :type keep_aspect_ratio: bool, optional
    :param fill_mode: Points outside the boundaries of the image are filled according to the given mode
                      (one of `{'constant', 'nearest', 'reflect', 'wrap'}`), defaults to ('nearest', 'nearest').
    :type fill_mode: Tuple[str, str], optional
    :param cval: Values used for points outside the boundaries of the image, defaults to (0., 0.).
    :type cval: Tuple[float, float], optional
    :param interp_order: Spline interpolation orders, defaults to (0, 0).
    :type interp_order: Tuple[int, int], optional
    """

    def __init__(self,
                 probability: float,
                 rotation: float = 0.,
                 translate: Tuple[float, float] = (0., 0.),
                 shear: float = 0.,
                 zoom: Tuple[float, float] = (1., 1.),
                 keep_aspect_ratio: bool = True,
                 fill_mode: Tuple[str, str] = ('nearest', 'nearest'),
                 cval: Tuple[float, float] = (0., 0.),
                 interp_order: Tuple[int, int] = (0, 0),
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self._args = locals()
        self._args.pop("self")

        if isinstance(translate, (float, int)):
            translate = [translate] * 2
        if isinstance(fill_mode, str):
            fill_mode = [fill_mode] * 2
        if isinstance(cval, (float, int)):
            cval = [cval] * 2
        if isinstance(interp_order, int):
            interp_order = [interp_order] * 2

        assert len(zoom) == 2

        self._probability = probability
        self._rotation = rotation
        self._translate = translate
        self._shear = shear
        self._zoom = zoom
        self._keep_aspect_ratio = keep_aspect_ratio
        self._fill_mode = fill_mode
        self._cval = cval
        self._interp_order = interp_order
        self._ndim = 2

    @staticmethod
    def apply_transform(x: np.ndarray,
                        transform_matrix: np.ndarray,
                        channel_axis: int = 0,
                        fill_mode: str = 'nearest',
                        cval: float = 0.,
                        interp_order: int = 0) -> np.ndarray:
        """
        Apply a transformation matrix to an image.

        :param x: An image (3D tensor).
        :type x: numpy.ndarray
        :param transform_matrix: A 3x3 transformation matrix.
        :type transform_matrix: numpy.ndarray
        :param channel_axis: Index of axis for channels. Defaults to 0.
        :type channel_axis: int, optional
        :param fill_mode: Points outside the boundaries of the image are filled according to the given mode.
                          Defaults to 'nearest'.
        :type fill_mode: str, optional
        :param cval: Value used for points outside the boundaries of the image. Defaults to 0.
        :type cval: float, optional
        :param interp_order: The order of the spline interpolation. Defaults to 0.
        :type interp_order: int, optional
        :return: A transformed image.
        :rtype: numpy.ndarray

        See also: :class:`~keras.preprocessing.image.ImageDataGenerator`
        """

        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=interp_order,  # NOTE: The order of the spline interpolation
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    @staticmethod
    def transform_matrix_offset_center(matrix: np.ndarray, x: int, y: int) -> np.ndarray:
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def zoom_matrix(zx, zy):
        matrix = np.array([[zx, 0, 0],
                           [0, zy, 0],
                           [0, 0, 1]])
        return matrix

    @staticmethod
    def rotation_matrix(theta):
        matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
        return matrix

    @staticmethod
    def translate_matrix(tx, ty):
        matrix = np.array([[1, 0, tx],
                           [0, 1, ty],
                           [0, 0, 1]])
        return matrix

    @staticmethod
    def shear_matrix(shear):
        matrix = np.array([[1, -np.sin(shear), 0],
                           [0, np.cos(shear), 0],
                           [0, 0, 1]])
        return matrix

    def affine(self, x: np.ndarray, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0, zx: float = 1,
               zy: float = 1,
               fill_mode: str = 'nearest', cval: float = 0., interp_order: int = 0) -> np.ndarray:
        shape = x.shape

        matrix = np.eye(3)
        if theta != 0:
            matrix = np.dot(matrix, self.rotation_matrix(theta))
        if tx != 0 or ty != 0:
            matrix = np.dot(matrix, self.translate_matrix(tx, ty))
        if shear != 0:
            matrix = np.dot(matrix, self.shear_matrix(shear))
        if zx != 1 or zy != 1:
            matrix = np.dot(matrix, self.zoom_matrix(zx, zy))

        if np.any(matrix != np.eye(3)):
            h, w = shape[_row_axis], shape[_col_axis]
            matrix = self.transform_matrix_offset_center(matrix, h, w)
            x = self.apply_transform(x, matrix, _channel_axis,
                                     fill_mode, cval, interp_order)
        return x

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x: Union[List[np.ndarray], None], y: Union[List[np.ndarray], None]) -> \
            Tuple[Union[List[np.ndarray], None], Union[List[np.ndarray], None]]:
        if self._rotation:
            theta = np.pi / 180 * \
                    np.random.uniform(-self._rotation, self._rotation)
        else:
            theta = 0

        if self._translate[0]:
            tx = np.random.uniform(-self._translate[0], self._translate[0])
        else:
            tx = 0

        if self._translate[1]:
            ty = np.random.uniform(-self._translate[1], self._translate[1])
        else:
            ty = 0

        if self._shear:
            shear = np.random.uniform(-self._shear, self._shear)
        else:
            shear = 0

        if self._zoom[0] == 1 and self._zoom[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self._zoom[0], self._zoom[1], 2)

        if self._keep_aspect_ratio:
            zy = zx

        fill_mode = self._fill_mode
        cval = self._cval
        interp_order = self._interp_order

        if np.random.random() < self._probability:
            if x is not None:
                x = [self.affine(x_i, theta, tx, ty, shear, zx, zy,
                                 fill_mode[0], cval[0], interp_order[0]) for x_i in x]
            if y is not None:
                y = [self.affine(y_i, theta, tx, ty, shear, zx, zy,
                                 fill_mode[1], cval[1], interp_order[1]) for y_i in y]
        return x, y


class Distort(Operation):
    """
    This class performs randomised, elastic distortions on images.
    """

    def __init__(self,
                 probability: float,
                 alpha: Tuple[float, float],
                 sigma: float,
                 order: Tuple[int, int] = (3, 0),
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._args = locals()
        self._args.pop("self")

        self._probability = probability
        self._alpha = alpha
        self._sigma = sigma
        self._order = order
        self._ndim = 2

    # Based on elastic_transform.py by fmder (https://gist.github.com/fmder/e28813c1e8721830ff9c)
    @staticmethod
    def random_distort(img, alpha, sigma, order):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        img_ = copy.deepcopy(img)
        shape = img_.shape
        img_ = np.squeeze(img_)

        dx = ndi.gaussian_filter((np.random.rand(shape[-2], shape[-1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndi.gaussian_filter((np.random.rand(shape[-2], shape[-1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[-2]), np.arange(shape[-1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        out = ndi.map_coordinates(img_, indices, order=order).reshape((shape[-2], shape[-1]))
        if len(shape) == 3:
            out = np.expand_dims(out, axis=0)

        return out

    @property
    def ndim(self) -> int:
        return self._ndim

    def apply_core(self, x, y):

        if np.random.random() < self._probability:
            alpha = np.random.uniform(self._alpha[0], self._alpha[1])
            if x is not None:
                x = [self.random_distort(x_i, alpha, self._sigma, self._order[0])
                     for x_i in x]
            if y is not None:
                y = [self.random_distort(y_i, alpha, self._sigma, self._order[1])
                     for y_i in y]

        return x, y
