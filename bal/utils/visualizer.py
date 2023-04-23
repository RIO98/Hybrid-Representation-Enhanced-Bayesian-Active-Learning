import numpy as np
import skimage.io
import skimage.util

from .vis2d.image_visualizer import ImageVisualizer
from .vis2d.labeled_image_visualizer import LabeledImageVisualizer
from .look_up_table.look_up_table import LookUpTable

_default_image_config = {
    'image_lut': LookUpTable('gray'),
    'image_clim': 'none',
}

_default_label_config = {
    'image_lut': LookUpTable('muscles'),
    'image_clim': 'none',
}

_default_labeled_config = {
    'image_lut': LookUpTable('gray'),
    'label_lut': LookUpTable('muscles'),
    'image_weight': 1.,
    'label_weight': 0.45,
    'image_clim': 'none',
    'label_indices': [i for i in range(1, 23)],
}


class Visualizer:

    def __init__(self,
                 image_config=_default_image_config,
                 label_config=_default_label_config,
                 labeled_config=_default_labeled_config):

        self.image_config = image_config
        self.label_config = label_config
        self.labeled_config = labeled_config

    def __call__(self, images, save_path):

        _image_arrays = []
        for i in range(len(images)):
            _image_arrays.append({
                'image': np.squeeze(images[i]['image'][0]),  # (h, w)
                'label': np.argmax(images[i]['label'][0], axis=0),  # (h, w)
                'gt': images[i]['gt'],  # (h, w)
            })

        self._vis_images = _image_arrays
        self._image_visualizer = ImageVisualizer(**self.image_config)
        self._label_visualizer = ImageVisualizer(**self.label_config)
        self._labeled_visualizer = LabeledImageVisualizer(**self.labeled_config)

        self.save_path = save_path
        self.create_montage()

    def create_montage(self):

        n_row = len(self._vis_images)
        n_col = 5  # image, label, overlaid, gt, gt_overlaid

        # fig, ax = plt.subplots(n_col, n_row, figsize=(8, 20))
        montage_images = []
        for n in range(n_row):
            image = self._vis_images[n]['image']
            label = self._vis_images[n]['label']
            gt = self._vis_images[n]['gt']

            vis_image = self._image_visualizer.visualize(image * 255)
            vis_label = self._label_visualizer.visualize(label)
            vis_gt = self._label_visualizer.visualize(gt)
            overlay_label = self._labeled_visualizer.visualize(image * 255, label)
            overlay_gt = self._labeled_visualizer.visualize(image * 255, gt)

            montage_images.append(vis_image[..., ::-1])
            montage_images.append(vis_label[..., ::-1])
            montage_images.append(overlay_label[..., ::-1])
            montage_images.append(vis_gt[..., ::-1])
            montage_images.append(overlay_gt[..., ::-1])

        montage = skimage.util.montage(montage_images, channel_axis=-1, grid_shape=(n_row, n_col))
        montage = montage.astype(np.uint8)
        skimage.io.imsave(self.save_path, montage)

    @property
    def vis_images(self):
        return self._vis_images

    # @property
    # def visualizer(self):
    # return self._visualizer
