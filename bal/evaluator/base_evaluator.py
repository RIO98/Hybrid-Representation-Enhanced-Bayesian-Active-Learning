import os
from abc import ABCMeta, abstractmethod
from glob import glob

import numpy as np
import pandas as pd
from medpy.metric import dc
from natsort import natsorted

from bal.data.io.image_file_handler import ImageFileHandler


class BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, pt: str, threshold: float,
                 test_root: str,
                 bank_root: str,
                 ref_root: str,
                 mode: str = "testing",
                 n_classes: int = 5):
        self.patient = pt
        self.mode = mode
        self.test_root = test_root
        self.bank_root = bank_root
        self.ref_root = ref_root
        self.threshold = threshold
        self.n_classes = n_classes

    @staticmethod
    def load_test_slices(paths, data_type=np.uint8):
        """Load test labels from the given paths."""
        lbls = []
        for path in paths:
            tmp_lbl, _ = ImageFileHandler.load_image(path)
            assert tmp_lbl.shape[-1] == 1, "Index not in the last dimension!"
            lbls.append(tmp_lbl)
        lbl_vol = np.asarray(lbls).astype(data_type).squeeze()

        return lbl_vol

    def calculate_dc(self):
        dice_coef = list()
        test_paths = natsorted(glob(os.path.join(self.test_root, self.patient, "image_*_pred.mha")))
        ref_paths = natsorted(glob(os.path.join(self.ref_root, self.patient, "label_*.mha")))
        assert len(test_paths) == len(ref_paths), f"Missing slices in case {self.patient}"
        print(f"Number of prediction slices for case {self.patient}: {len(test_paths)}")

        test_lbl, ref_lbl = self.load_test_slices(test_paths), self.load_test_slices(ref_paths)
        lbl_class = len(np.unique(ref_lbl))  # label class plus background
        assert lbl_class == self.n_classes, f"Missing roi in case {self.patient}"
        for i in range(1, lbl_class):
            dice_coef.append(dc((test_lbl == i), (ref_lbl == i)))

        dice_coef.append(np.average(dice_coef))

        return dice_coef

    @abstractmethod
    def get_uncertainty(self):
        raise NotImplementedError("get_uncertainty() is not implemented!")

    def run(self):
        """Run the active iterator."""
        if self.mode == "testing":
            dice = self.calculate_dc()
            return pd.DataFrame(
                data=[dice],
                columns=list(range(self.n_classes))[1:] + ["all"],
                index=[self.patient.replace("_label", "")]
            )
        elif self.mode == "databank":
            uncert = self.get_uncertainty()
            return pd.DataFrame(
                data=[uncert],
                columns=["all"],
                index=[self.patient.replace("_label", "")]
            )
