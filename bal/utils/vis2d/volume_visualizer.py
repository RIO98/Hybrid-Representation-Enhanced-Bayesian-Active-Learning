import itertools

import cv2
import numpy as np

from bal.utils.vis2d.image_visualizer import ImageVisualizer


class VolumeVisualizer(ImageVisualizer):

    def __init__(self,
                 image_lut,
                 image_clim='min-max',
                 dim_order='zyx',
                 random_seed=0):

        super(VolumeVisualizer, self).__init__(
            image_lut=image_lut,
            image_clim=image_clim,
        )

        self.slice_indices = None
        self.bbox = None
        self.img_vol = None
        self.spacing = None
        np.random.seed(random_seed)
        self.dim_order = dim_order

    @staticmethod
    def get_bbox(img):

        n_dims = img.ndim
        out = []
        for ax in itertools.combinations(range(n_dims), n_dims - 1):
            nonzero = np.any(img, axis=ax)
            out.extend(np.where(nonzero)[0][[0, -1]])
        return tuple(out)

    def set_data(self, img_vol, spacing, crop=True):

        self.img_vol = img_vol
        self.spacing = spacing

        if self.dim_order == 'xyz':
            self.img_vol = img_vol.transpose(2, 1, 0)

        self.img_vol = self._clip_range(self.img_vol)

        if crop:
            self.bbox = self.get_bbox(self.img_vol > 0)
        else:
            vol_shape = self.img_vol.shape
            self.bbox = [0, vol_shape[2], 0, vol_shape[1], 0, vol_shape[0]]

    def _select_slices(self,
                       plane='axial',
                       selection='equal',
                       n_slices=None):

        assert selection in ['random', 'all', 'equal'], \
            'Invalid selection method was indicated.({})'.format(selection)
        assert plane in ['axial', 'coronal', 'sagittal'], \
            'Invalid plane was indicated.({})'.format(plane)

        shape = self.img_vol.shape
        if plane == 'axial':
            min_idx, max_idx = 0, shape[0] - 1
        elif plane == 'coronal':
            min_idx, max_idx = 0, shape[1] - 1
        elif plane == 'sagittal':
            min_idx, max_idx = 0, shape[2] - 1

        if selection == 'random':
            if n_slices is None:
                raise ValueError('The number of slices must be '
                                 'indicated when you use `random` selection.')
            slice_indices = sorted(list(np.random.randint(min_idx, max_idx, n_slices)))
        elif selection == 'all':
            slice_indices = [i for i in range(min_idx, max_idx + 1)]
        elif selection == 'equal':
            interval = (max_idx - min_idx) // n_slices
            slice_indices = [i for i in range(min_idx, max_idx + 1, interval)][:n_slices]

        return slice_indices

    def _get_slices(self, vol, plane):

        images = []

        for idx in self.slice_indices:
            if plane == 'axial':
                images.append(vol[idx, :, :])
            if plane == 'coronal':
                images.append(vol[:, idx, :])
            if plane == 'sagittal':
                images.append(vol[:, :, idx])

        return images

    def resize_image(self, image, plane, interp='nearest'):

        if interp == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif interp == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError('Invalid interpolation method.')

        height, width = image.shape[0], image.shape[1]

        if plane == 'axial':
            ratio = self.spacing[2] / self.spacing[1]
        if plane == 'coronal':
            ratio = self.spacing[0] / self.spacing[2]
        if plane == 'sagittal':
            ratio = self.spacing[0] / self.spacing[1]

        dsize = (int(width), int(height * ratio))
        resized_image = cv2.resize(image, dsize, interpolation=interpolation)
        return resized_image

    def visualize(self,
                  plane,
                  selection=None,
                  n_slices=None,
                  slice_indices=None):

        if slice_indices is not None:
            self.slice_indices = slice_indices
        else:
            self.slice_indices = self._select_slices(plane, selection, n_slices)

        images = self._get_slices(self.img_vol, plane)

        images = [self._set_color(img, self.image_lut) for img in images]

        images = [self.resize_image(img, plane, interp='linear') for img in images]

        images = np.array(images)
        return images
