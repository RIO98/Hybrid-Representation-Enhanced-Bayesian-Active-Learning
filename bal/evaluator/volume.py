import os
from abc import ABC
from glob import glob
from natsort import natsorted

import numpy as np

from .base_evaluator import BaseEvaluator
from ..selector.volume import RandomSelector, UncertaintySelector, DiversitySelector, DensitySelector, HybridSelector

ACTIVE_SELECTORS = {
    "random": RandomSelector,
    "unc": UncertaintySelector,
    "simi": DensitySelector,
    "mi": DiversitySelector,
    "simi_mi": HybridSelector
}


class ActiveEvaluator(BaseEvaluator, ABC):
    def __init__(self, pt: str,
                 threshold: float,
                 test_root: str,
                 bank_root: str,
                 ref_root: str,
                 mode: str = "testing",
                 n_classes: int = 5):
        super(ActiveEvaluator, self).__init__(pt, threshold, test_root, bank_root, ref_root, mode, n_classes)

    def get_uncertainty(self):
        bank_uncert_paths = natsorted(glob(os.path.join(self.bank_root, self.patient, "image_*_uncert.mha")))
        print("Number of uncertainty slices: {}".format(len(bank_uncert_paths)))
        uncert = self.load_test_slices(bank_uncert_paths, data_type=np.float32)
        mask = (uncert >= self.threshold).astype(int)

        return np.sum(uncert * mask) / np.count_nonzero(mask) if np.count_nonzero(mask) != 0 else 0
