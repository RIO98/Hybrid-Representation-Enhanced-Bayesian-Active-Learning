import numpy as np
import cv2

from bal.utils.remove_island import remove_island


class ImageProcess:
    @staticmethod
    def pad_to_target_shape(image, target_shape):
        assert len(image.shape) == len(target_shape), "Image shape and target shape must have the same dimension."
        pad_width = tuple(((target - image_dim) // 2,
                           (target - image_dim + 1) // 2)
                          for image_dim, target in zip(image.shape, target_shape))

        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

        return padded_image

    @staticmethod
    def half_size_image(image: np.ndarray, method) -> np.ndarray:
        return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=method)

    @staticmethod
    def rm_is(img: np.ndarray) -> np.ndarray:
        """
        Remove isolated objects from a binary/multi-class label.

        :param img: Input a binary/multi-class label as a numpy array.
        :type img: np.ndarray
        :return: Binary/multi-class label with isolated objects removed, as a numpy array.
        :rtype: np.ndarray
        """
        return remove_island(img, only_largest=True)

    @staticmethod
    def set_window(x: np.ndarray, ww: int = 500, wl: int = 100) -> np.ndarray:
        """
        Set window level and width of a 3D volume.

        :param x: Input 3D volume as a numpy array.
        :type x: np.ndarray
        :param ww: Window width in Hounsfield units (default=500).
        :type ww: int
        :param wl: Window level in Hounsfield units (default=100).
        :type wl: int
        """

        vmax = wl + ww // 2
        vmin = wl - ww // 2
        x[x > vmax] = vmax
        x[x < vmin] = vmin
        x = (255 / (vmax - vmin)) * (x - vmin)
        x = x.astype('uint8')
        return x

    @staticmethod
    def clip_and_norm(x: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
        x = np.clip(x, clip_min, clip_max)
        x = (x - clip_min) / (clip_max - clip_min)  # [0, 1]

        return x
