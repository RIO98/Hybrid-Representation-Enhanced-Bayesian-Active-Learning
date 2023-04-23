from typing import Optional

import numpy as np
import torch


def get_class_weight(cls_weights: str) -> Optional[torch.Tensor]:
    if cls_weights == 'none':
        return None

    # Dictionary mapping class weights string to the corresponding tensor
    class_weights_map = {
        'tri': lambda: torch.tensor([1.0, 1.5, 1.5, 1.5]),
        'gluteus': lambda: torch.tensor([1.0, 1.5, 1.5, 1.5]),
        'quad': lambda: torch.tensor([1.0, 1.5, 1.5, 1.5, 1.5]),
        'binary': lambda: torch.tensor([1.0, 1.5]),
    }

    if cls_weights in class_weights_map:
        return class_weights_map[cls_weights]()
    else:
        raise NotImplementedError(f"Invalid class weights type: {cls_weights}")
