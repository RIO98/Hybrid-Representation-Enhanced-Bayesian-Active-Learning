import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for datasets.
    """

    def __init__(
            self,
            root: str,
            exts: Dict[str, str],
            patients: List[str],
            classes: Optional[List[str]] = None,
            dtypes: Optional[Dict[str, np.dtype]] = None,
            filenames: Optional[Dict[str, str]] = None,
            normalizer: Optional[Any] = None,
            augmentor: Optional[Any] = None
    ):
        # Initialize
        files = OrderedDict()
        file_sizes = []

        for key in filenames.keys():
            files[key] = []
            for s in tqdm.tqdm(patients, desc='Collecting %s slices' % key, ncols=80):
                if key == 'image':
                    files[key].extend(glob(filenames[key].format(root=root, patients=s)))
                else:
                    files[key].extend(
                        glob(filenames[key].format(root=root, patients=s.replace('image', 'label'))))

            if len(files[key]) == 0:
                warnings.warn('%s files are not found.. ' % key)
            file_sizes.append(len(files[key]))

        self._root = root
        self._ext = exts
        self._patients = patients
        self._classes = classes
        self._dtypes = dtypes
        self._filenames = filenames
        self._normalizer = normalizer
        self._augmentor = augmentor
        self._files = files

    def __len__(self) -> int:
        key = list(self._files.keys())[0]
        return len(self._files[key])

    def __getitem__(self, idx: int) -> Tuple:
        return self.get_example(idx)

    @property
    def classes(self) -> Optional[List[str]]:
        return self._classes

    @property
    def n_classes(self) -> Optional[int]:
        if self.classes is None:
            return None
        return len(self.classes)

    @property
    def files(self) -> OrderedDict:
        return self._files

    @classmethod
    @abstractmethod
    def get_example(cls, i: int) -> Tuple:
        """
        Get an example from the dataset given its index.

        :param i: The index of the example.
        :return: A tuple containing the image and label for the given index.
        """

        raise NotImplementedError()

    @abstractmethod
    def __copy__(self) -> 'BaseDataset':
        """Copy the class instance."""
        raise NotImplementedError()


if __name__ == '__main__':
    base_dataset = BaseDataset()
