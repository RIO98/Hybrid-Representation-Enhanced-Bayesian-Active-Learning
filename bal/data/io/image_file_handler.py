import os
from typing import Union, Tuple, List

import cv2
import numpy as np

from .meta_image import MetaImage


class ImageFileHandler:
    """
    Class for handling image file I/O.
    """

    @staticmethod
    def load_image(filename: str, with_offset: bool = False) -> Union[
        Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]], Tuple[
            np.ndarray, Union[np.ndarray, None]]]:
        """
        Loads an image from a file and returns the image, its spacing, and optionally its offset.
        """
        ext = ImageFileHandler._get_extension(filename)

        if ext in ('.mha', '.mhd'):
            img, img_header = MetaImage.read(filename)
            spacing = img_header['ElementSpacing']
            if with_offset:
                offset = img_header['Offset']
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
        elif ext in ('.png', '.jpg', '.bmp'):
            img = cv2.imread(filename)
            spacing = None
            if with_offset:
                offset = None
        else:
            raise NotImplementedError()

        if with_offset:
            return img, spacing, offset
        else:
            return img, spacing

    @staticmethod
    def save_image(filename: str, image: np.ndarray, spacing: Union[Tuple, List] = None,
                   offset: Union[Tuple, List] = None) -> None:
        """
        Saves an image to a file with the provided spacing and optionally its offset.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ext = ImageFileHandler._get_extension(filename)

        if ext in ('.mha', '.mhd'):
            header = ImageFileHandler._create_header(image, spacing, offset)
            if image.ndim == 3:
                image = image.transpose((2, 0, 1))
            MetaImage.write(filename, image, header)
        elif ext in ('.png', '.jpg', '.bmp'):
            cv2.imwrite(filename, image)
        else:
            raise NotImplementedError()

    @staticmethod
    def _get_extension(filename: str) -> str:
        _, ext = os.path.splitext(os.path.basename(filename))
        return ext

    @staticmethod
    def _create_header(image: np.ndarray, spacing: Union[Tuple, List], offset: Union[Tuple, List]) -> dict:
        header = {}
        if spacing is not None:
            header['ElementSpacing'] = spacing

        if offset is not None:
            header['Offset'] = offset

        if image.ndim == 2:
            header['TransformMatrix'] = '1 0 0 1'
            header['CenterOfRotation'] = '0 0'
        elif image.ndim == 3:
            header['TransformMatrix'] = '1 0 0 0 1 0 0 0 1'
            header['CenterOfRotation'] = '0 0 0'
        else:
            raise NotImplementedError()

        return header
