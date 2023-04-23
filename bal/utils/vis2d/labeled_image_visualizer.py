import numpy as np

from .image_visualizer import ImageVisualizer
from .labeled_visualizer import LabeledVisualizer
from bal.utils.look_up_table.look_up_table import LookUpTable


class LabeledImageVisualizer(ImageVisualizer, LabeledVisualizer):
    """
    This class provides methods to visualize labeled images/contours and overlay them on original images.
    """

    def __init__(self,
                 image_lut: LookUpTable = None,
                 label_lut: LookUpTable = None,
                 image_weight: int = 1,
                 label_weight: int = 1,
                 image_clim: str = 'min-max',
                 label_indices: list = None,
                 contour_indices: list = None,
                 contour_width: int = None):
        
        """
        Initializes LabeledImageVisualizer object with specified look-up tables, image and label weights, color range,
        label indices, contour indices and contour width.

        :param image_lut: LookUpTable object to apply color mapping to images.
        :type image_lut: LookUpTable, optional
        :param label_lut: LookUpTable object to apply color mapping to labels.
        :type label_lut: LookUpTable, optional
        :param image_weight: Weight to apply to the original image while overlaying the labels.
        :type image_weight: int, optional
        :param label_weight: Weight to apply to the labeled image while overlaying it on the original image.
        :type label_weight: int, optional
        :param image_clim: Color range to map the image to. Can be 'min-max', 'none', or a tuple of (min, max).
        :type image_clim: str or tuple, optional
        :param label_indices: List of label indices to visualize.
        :type label_indices: list, optional
        :param contour_indices: List of label indices to generate contours for.
        :type contour_indices: list, optional
        :param contour_width: Number of erosion iterations, negatively correlated to the width of the contours to be generated.
        :type contour_width: int, optional
        """

        super(LabeledImageVisualizer, self).__init__(
            image_lut=image_lut,
            image_clim=image_clim,
        )

        super(ImageVisualizer, self).__init__(
            label_indices=label_indices,
            contour_indices=contour_indices,
            contour_width=contour_width,
        )

        assert isinstance(label_lut, LookUpTable) or (label_lut is None)

        self.label_lut = label_lut
        self.image_weight = image_weight
        self.label_weight = label_weight

    def visualize(self, image: np.ndarray, label: np.ndarray) -> np.ndarray:

        assert len(label.shape) == 2, 'Invalid label shape: {}'.format(label.shape)

        label_tmp = label.copy()

        if self.label_indices is None:
            a, b = np.min(label), np.max(label)
            self.label_indices = [i for i in range(a, b + 1)]

        if self.contour_indices is not None:
            label_tmp = self._gen_contour(label_tmp)

        image_tmp = image.copy()

        if len(image.shape) == 2:
            image_tmp = self._clip_range(image_tmp)

        if self.label_weight == 1:
            for i in self.label_indices:
                image_tmp[label_tmp == i] = 0

        image_color = self._set_color(image_tmp, self.image_lut)
        label_color = self._set_color(label_tmp, self.label_lut)

        overlaid = self._overlay(image_color, label_color)

        return overlaid
