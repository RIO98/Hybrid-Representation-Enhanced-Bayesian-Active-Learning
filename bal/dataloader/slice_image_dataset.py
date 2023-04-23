from __future__ import annotations

from collections import OrderedDict
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy import ndarray

from bal.dataloader.base_dataset import BaseDataset
from bal.data.io.image_file_handler import ImageFileHandler

_supported_filetypes = [
    'image',
    'label',
    'mask',
]

_default_dtypes = OrderedDict({
    'image': np.float32,
    'label': np.int32,
    'mask': np.uint8,
})

_default_filenames = OrderedDict({
    'image': '{root}/{slice}',
    'label': '{root}/{slice}',
    'mask': '{root}/{slice}',
})

_default_mask_cvals = OrderedDict({
    'image': 0,
    'label': 0,
})

_channel_axis = 0


def _inspect_n_args(func) -> int:
    sig = signature(func)
    return len(sig.parameters)


class SliceImageDataset(BaseDataset):

    def __init__(
            self,
            root: str,
            exts: Dict[str, str],
            patients: List[str],
            classes: Optional[List[str]] = None,
            dtypes: Dict[str, np.dtype] = _default_dtypes,
            filenames: Dict[str, str] = _default_filenames,
            normalizer: Optional[Any] = None,
            augmentor: Optional[Any] = None
    ):

        super(SliceImageDataset, self).__init__(root, exts, patients, classes, dtypes, filenames, normalizer, augmentor)

        assert (isinstance(patients, (list, np.ndarray)))

        self._args = locals()
        self._args.pop('self')

        self.root = root
        if isinstance(patients, np.ndarray):
            patients = patients.tolist()

        self.exts = exts
        self.patients = patients
        self.dtypes = dtypes
        self.augmentor = augmentor
        self.normalizer = normalizer

    @staticmethod
    def check_input(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # image
        if image.ndim not in [2, 3]:
            raise ValueError()
        elif image.ndim == 2:
            image = image[:, :, np.newaxis]  # channel-last

        # label
        if image.ndim not in [2, 3]:
            raise ValueError()
        elif label.ndim == 3:
            if label.shape[-1] in [1, 3]:
                label = label[:, :, 0]  # reduce
            else:
                raise ValueError()

        return image, label

    def normalize(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # reshape
        if x.ndim == 2:
            x = x[np.newaxis]
        elif x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))  # [c, w, h]

        if y is not None:
            # NOTE: assume that `y` is a categorical label
            if y.dtype in [np.int32, np.int64]:
                if y.ndim == 3:
                    if y.shape[-1] in [1, 3]:
                        y = y[:, :, 0]  # NOTE: ad-hoc
                    else:
                        pass

            # NOTE: assume that `y` is continuous label (e.g., heatmap)
            elif y.dtype in [np.float32, np.float64]:
                if y.ndim == 2:
                    y = y[np.newaxis]
                elif y.ndim == 3:
                    y = np.transpose(y, (2, 0, 1))  # [c, w, h]

            else:
                raise NotImplementedError('unsupported data type..')

        # normalizer
        if self.normalizer is not None:
            if _inspect_n_args(self.normalizer) == 2:
                x, y = self.normalizer(x, y)
            else:
                x = self.normalizer(x)

        return x, y

    def denormalize(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError()

    def load_images(self, i: int) -> tuple[dict[Any, tuple[ndarray, str, str] | tuple[ndarray, str] | Any], dict[
        Any, tuple[ndarray, str, str] | tuple[ndarray, str]]]:
        images, spacings = {}, {}
        for key in self.files.keys():
            images[key], spacings[key] = ImageFileHandler.load_image(self.files[key][i])
            images[key] = images[key].astype(self.dtypes[key])

        return images, spacings

    def get_example(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load images
        images, _ = self.load_images(i)
        image = images['image']
        label = images['label']

        # check
        image, label = SliceImageDataset.check_input(image, label)

        # normalize
        image, label = self.normalize(image, label)

        # augmentation
        if self.augmentor is not None:
            if _inspect_n_args(self.augmentor) == 2:
                image, label = self.augmentor(image, label)
            else:
                image = self.augmentor(image)

        image = torch.from_numpy(image)
        label = torch.from_numpy(np.ascontiguousarray(label))

        return image, label

    def __copy__(self):
        return SliceImageDataset(self._root,
                                 self._ext,
                                 self._patients,
                                 self._classes,
                                 self._dtypes,
                                 self._filenames,
                                 self._normalizer,
                                 self._augmentor)
