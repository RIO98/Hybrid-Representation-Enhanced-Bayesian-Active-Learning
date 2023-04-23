import numpy as np

from .labeled_visualizer import LabeledVisualizer
from bal.utils.vis2d.volume_visualizer import VolumeVisualizer
from bal.utils.look_up_table.look_up_table import LookUpTable


class LabeledVolumeVisualizer(VolumeVisualizer, LabeledVisualizer):

    def __init__(self,
                 image_lut=None,
                 label_lut=None,
                 image_weight=1,
                 label_weight=1,
                 image_clim='min-max',
                 label_indices=None,
                 contour_indices=None,
                 contour_width=None,
                 dim_order='zyx',
                 random_seed=0,
                 ):

        super(LabeledVolumeVisualizer, self).__init__(
            image_lut=image_lut,
            image_clim=image_clim,
        )
        LabeledVisualizer.__init__(
            self,
            label_indices=label_indices,
            contour_indices=contour_indices,
            contour_width=contour_width,
        )

        self.lbl_vol = None
        np.random.seed(random_seed)

        assert isinstance(label_lut, LookUpTable) or (label_lut is None)

        self.label_lut = label_lut
        self.image_weight = image_weight
        self.label_weight = label_weight
        self.dim_order = dim_order

    def set_data(self, img_vol, lbl_vol, spacing, crop=True):

        self.img_vol = img_vol
        self.lbl_vol = lbl_vol
        self.spacing = spacing

        if self.dim_order == 'xyz':
            self.img_vol = img_vol.transpose(2, 1, 0)
            self.lbl_vol = lbl_vol.transpose(2, 1, 0)

        self._erase_labels()
        self.img_vol = self._clip_range(self.img_vol)

        if crop:
            self.bbox = self.get_bbox(self.lbl_vol)
        else:
            vol_shape = self.lbl_vol.shape
            self.bbox = [0, vol_shape[2], 0, vol_shape[1], 0, vol_shape[0]]

    def _erase_labels(self):

        a, b = np.max(self.lbl_vol), np.max(self.label_indices)
        for i in range(np.max((a, b)) + 1):
            if i not in self.label_indices:
                self.lbl_vol[self.lbl_vol == i] = 0

    def visualize(self, plane, selection=None, n_slices=None, slice_indices=None):

        if self.label_indices is None:
            a, b = np.min(self.lbl_vol), np.max(self.lbl_vol)
            self.label_indices = [i for i in range(a, b + 1)]

        if slice_indices is not None:
            self.slice_indices = slice_indices
        else:
            self.slice_indices = self._select_slices(plane, selection, n_slices)

        img_vol = self.img_vol.copy()
        lbl_vol = self.lbl_vol.copy()

        if self.contour_indices is not None:
            lbl_vol = self._gen_contour(lbl_vol)

        if self.label_weight == 1:
            for i in self.label_indices:
                img_vol[lbl_vol == i] = 0

        images = self._get_slices(img_vol, plane)
        labels = self._get_slices(lbl_vol, plane)

        images = [self._set_color(img, self.image_lut) for img in images]
        labels = [self._set_color(lbl, self.label_lut) for lbl in labels]

        images = [self.resize_image(img, plane, interp='linear') for img in images]
        labels = [self.resize_image(lbl, plane, interp='nearest') for lbl in labels]

        overlaid = [self._overlay(img, lbl) for img, lbl in zip(images, labels)]

        overlaid = np.array(overlaid)

        return overlaid
