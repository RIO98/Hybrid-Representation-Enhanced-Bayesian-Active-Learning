import cv2
import numpy as np


class LabeledVisualizer:
    """
    This class provides methods to generate RGB label images or contours.
    """

    def __init__(self,
                 label_indices: list = None,
                 contour_indices: list = None,
                 contour_width: int = None):

        """
        Initializes LabeledVisualizer object with specified label and contour indices and contour width.

        :param label_indices: List of label indices to visualize.
        :type label_indices: list, optional
        :param contour_indices: List of label indices to generate contours for.
        :type contour_indices: list, optional
        :param contour_width: Number of erosion iterations, negatively correlated to the width of the contours to be generated.
        :type contour_width: int, optional
        """

        self.label_indices = label_indices
        self.contour_indices = contour_indices
        self.contour_width = contour_width

        if self.contour_indices is not None:
            assert self.contour_width > 0, \
                'Please set `contour_width` when you use the contour visualization.'

    @staticmethod
    def extract_contour(label: np.ndarray, iterations: int) -> np.ndarray:

        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        eroded = cv2.erode(label, kernel, iterations=iterations)
        contour = label - eroded
        return contour

    def _gen_contour(self, label: np.ndarray) -> np.ndarray:

        for i in self.contour_indices:
            l = label.copy() == i
            c = self.extract_contour(l.astype(np.uint8), self.contour_width)
            label[l] = 0
            label[c > 0] = i
        return label

    def _overlay(self, image: np.ndarray, label: np.ndarray) -> np.ndarray:

        # overlaid = cv2.addWeighted(image, self.image_weight, label, self.label_weight, gamma=0)
        overlaid = np.where(label >= 0,
                            image * (1 - self.label_weight) + label * self.label_weight,
                            image)
        return overlaid

    def visualize(self, image: np.ndarray, label: np.ndarray):
        raise NotImplementedError()
