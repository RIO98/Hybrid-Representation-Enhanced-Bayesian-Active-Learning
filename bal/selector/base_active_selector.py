import multiprocessing as mp
import os
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

from bal.data.io.image_file_handler import ImageFileHandler
from bal.utils.image_processor import ImageProcess
from bal.utils.utils import multiprocess_agent


class BaseActiveSelector(metaclass=ABCMeta):
    def __init__(self,
                 file_root: str,
                 train_paths: list,
                 bank_paths: list,
                 n_outputs: int,
                 clip: bool = False,
                 img_vmin: int = 0,
                 img_vmax: int = 1000):
        self.file_root = file_root
        self.train_paths = train_paths
        self.bank_paths = bank_paths
        self.train_data = None
        self.bank_data = None
        self.n_outputs = n_outputs
        self.clip = clip
        self.indices = []
        self.img_vmin = img_vmin
        self.img_vmax = img_vmax
        self.n_workers = mp.cpu_count() - 1

    @staticmethod
    def min_max(x):
        min_x = np.min(x)
        max_x = np.max(x)
        return (x - min_x) / (max_x - min_x) if (max_x - min_x) != 0 else np.zeros(x.shape, dtype=x.dtype)

    @staticmethod
    def mutual_info(img1, img2):
        """
        Calculate the mutual information (image similarity) between img1 and img2
        :param img1: np.ndarray
        :param img2: np.ndarray
        :return: float
        """
        hgram = np.histogram2d(img1.ravel(), img2.ravel(), 20)[0]
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1)  # marginal for x over y
        py = np.sum(pxy, axis=0)  # marginal for y over x
        px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
        nzs = pxy > 0

        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    @staticmethod
    def init_pool(data_pack):
        global shared_pack
        shared_pack = data_pack

    @staticmethod
    def vec_reader(idx):
        img = np.squeeze(
            ImageFileHandler.load_image(os.path.join(shared_pack.file_root,
                                                     shared_pack.paths[idx]))[0]).astype(np.float16)
        if shared_pack.clip:
            img = ImageProcess.clip_and_norm(img, shared_pack.img_vmin, shared_pack.img_vmax)

        return img.flatten()

    def read_vector(self):
        DataPack = namedtuple("DataPack", ["file_root", "paths", "clip", "img_vmin", "img_vmax"])
        if len(self.train_paths) != 0:
            args = [(i,) for i in range(len(self.train_paths))]
            train_data_pack = DataPack(self.file_root,
                                       self.train_paths,
                                       self.clip,
                                       self.img_vmin,
                                       self.img_vmax)
            out = multiprocess_agent(self.vec_reader,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(train_data_pack,),
                                     msg="Load training data")

            self.train_data = np.vstack(out)

        if len(self.bank_paths) != 0:
            args = [(i,) for i in range(len(self.bank_paths))]
            bank_data_pack = DataPack(self.file_root,
                                      self.bank_paths,
                                      self.clip,
                                      self.img_vmin,
                                      self.img_vmax)
            out = multiprocess_agent(self.vec_reader,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(bank_data_pack,),
                                     msg="Load unlabeled data")

            self.bank_data = np.vstack(out)

        print("Data loaded!")

    @staticmethod
    @abstractmethod
    def core_iter(entity_id):
        pass

    @abstractmethod
    def apply(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
