from typing import Any
from ..augmentor.data_augmentor import DataAugmentor


class Normalizer(DataAugmentor):

    def apply(self, x: Any) -> Any:
        x, is_expanded_x = self.preprocess(x)

        for op in self._operations:
            x = op.apply(x)

        x = self.postprocess(x, is_expanded_x)

        return x

    def __call__(self, x: Any) -> Any:
        return self.apply(x)
