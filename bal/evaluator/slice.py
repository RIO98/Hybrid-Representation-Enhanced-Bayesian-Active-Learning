import os
from abc import ABC

import numpy as np

from bal.data.io.image_file_handler import ImageFileHandler
from .base_evaluator import BaseEvaluator
from ..selector.slice import RandomSelector, UncertaintySelector, DiversitySelector, DensitySelector, HybridSelector

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
        bank_uncert_pth = os.path.join(self.bank_root, self.patient.replace('label', 'uncert'))
        bank_uncert, _ = ImageFileHandler.load_image(bank_uncert_pth)
        mask = (bank_uncert >= self.threshold).astype(int)

        return np.sum(bank_uncert * mask) / np.count_nonzero(mask) if np.count_nonzero(mask) != 0 else 0
