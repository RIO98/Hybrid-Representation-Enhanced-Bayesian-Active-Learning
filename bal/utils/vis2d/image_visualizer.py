import cv2
import numpy as np

from bal.utils.look_up_table.look_up_table import LookUpTable


class ImageVisualizer:
    """
    This class provides methods to visualize images with specified look-up table and color range.
    """

    def __init__(self, image_lut: LookUpTable = None, image_clim: str = 'min-max'):
        """
        Initializes ImageVisualizer object with specified look-up table and color range.

        :param image_lut: LookUpTable object to apply color mapping to grayscale images.
        :type image_lut: LookUpTable, optional
        :param image_clim: Color range to map the image to. Can be 'min-max', 'none', or a tuple of (min, max).
        :type image_clim: str or tuple, optional
        """

        assert isinstance(image_lut, LookUpTable) or (image_lut is None)

        self.image_lut = image_lut
        self.image_clim = image_clim

    @staticmethod
    def cast(image: np.ndarray) -> np.ndarray:

        if image.dtype != np.uint8:
            return image.astype(np.uint8)
        else:
            return image

    def _clip_range(self, image: np.ndarray) -> np.ndarray:

        if isinstance(self.image_clim, str):
            if self.image_clim == 'min-max':
                self.image_clim = (np.min(image), np.max(image))
            elif self.image_clim == 'none':
                return image
            else:
                raise NotImplementedError('unsupported image_clim type..')

        assert isinstance(self.image_clim, (list, tuple))

        image = np.clip(image, self.image_clim[0], self.image_clim[1])
        image = (255 / (self.image_clim[1] - self.image_clim[0])) * (image - self.image_clim[0])
        return image

    def _set_color(self, image: np.ndarray, image_lut: LookUpTable) -> np.ndarray:
        """
        Sets the color of the image/label using the specified look-up table.

        :param image: Image/label to set color for.
        :type image: np.ndarray
        :param image_lut: LookUpTable object to apply color/grayscale mapping.
        :type image_lut: LookUpTable
        :return: RGB image/label.
        :rtype: np.ndarray
        """

        image = self.cast(image)

        im_shape = image.shape

        if len(im_shape) == 2:  # 2D image/label
            cmap = image_lut([i for i in range(256)])[:, :3]  # Take only RGB channels
            assert np.max(image) <= len(cmap)

            cmap = 255. * cmap.copy()
            cmap = cmap.astype(np.uint8)
            cmap256 = np.zeros((256, 3), np.uint8)
            cmap256[:len(cmap)] = cmap

            im_r = cv2.LUT(image, cmap256[:, 2])  # NOTE: Due to OpenCV's B-G-R order
            im_g = cv2.LUT(image, cmap256[:, 1])  # NOTE: Due to OpenCV's B-G-R order
            im_b = cv2.LUT(image, cmap256[:, 0])  # NOTE: Due to OpenCV's B-G-R order

            image_color = cv2.merge((im_r, im_g, im_b))

        elif len(im_shape) == 3:  # RGB image
            image_color = image

        else:
            raise ValueError('Invalid image shape. ({})'.format(im_shape))

        return image_color

    def visualize(self, image: np.ndarray) -> np.ndarray:

        image = self._clip_range(image)
        image_color = self._set_color(image, self.image_lut)

        return image_color
