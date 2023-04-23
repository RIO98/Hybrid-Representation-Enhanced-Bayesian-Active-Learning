import multiprocessing as mp
import os
from collections import namedtuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bal.data.io.image_file_handler import ImageFileHandler
from bal.utils.image_processor import ImageProcess
from bal.utils import multiprocess_agent


class VolumeCluster:
    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list,
                 method: str = "simi", coef: float = 0.5, clip=False, img_vmin=0, img_vmax=1000, uncertainty=None):
        self.file_root = file_root
        self.method = method
        self.uncertainty = uncertainty
        self.train_paths = train_paths
        self.bank_paths = bank_paths
        self.buffer_cases = buffer_cases
        self.n_outputs = n_outputs
        self.index = []
        self.train_data = None
        self.bank_data = None
        self.coef = coef
        self.clip = clip
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
            ImageFileHandler.load_image(os.path.join(shared_pack.file_root, shared_pack.paths[idx]))[0]).astype(
            np.float16)
        if shared_pack.clip:
            img = ImageProcess.clip_and_norm(img, shared_pack.img_vmin, shared_pack.img_vmax)

        return img.flatten()

    def read_vector(self):
        DataPack = namedtuple("DataPack", ["file_root", "paths", "clip", "img_vmin", "img_vmax"])
        if len(self.train_paths) != 0:
            args = [(i,) for i in range(len(self.train_paths))]
            train_data_pack = DataPack(self.file_root, self.train_paths, self.clip, self.img_vmin, self.img_vmax)

            # with mp.Pool(self.n_workers, initializer=self.init_pool, initargs=(train_data_pack,)) as p:
            #     out = p.starmap(self.vec_reader, args)
            out = multiprocess_agent(self.vec_reader,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(train_data_pack,),
                                     msg="Load training data")

            self.train_data = np.vstack(out)

        if len(self.bank_paths) != 0:
            args = [(i,) for i in range(len(self.bank_paths))]
            bank_data_pack = DataPack(self.file_root, self.bank_paths, self.clip, self.img_vmin, self.img_vmax)

            # with mp.Pool(self.n_workers, initializer=self.init_pool, initargs=(bank_data_pack,)) as p:
            #     out = p.starmap(self.vec_reader, args)
            out = multiprocess_agent(self.vec_reader,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(bank_data_pack,),
                                     msg="Load unlabeled data")

            self.bank_data = np.vstack(out)

        print("Data loaded!")

    @staticmethod
    def simi_iter(case_n):
        img_indices = [i for i, pth in enumerate(shared_pack.bank_paths) if case_n in pth]
        buffer_data = shared_pack.bank_data[img_indices]
        tmp_simi_mat = cosine_similarity(buffer_data, shared_pack.bank_data)
        tmp_simi_mat[tmp_simi_mat < 0.] = 0.
        return np.mean(np.sum(tmp_simi_mat, axis=1))

    def simi(self, args):
        bank_pack = namedtuple("DataPack", ["bank_paths", "bank_data"])(self.bank_paths, self.bank_data)

        simi_score = multiprocess_agent(self.simi_iter,
                                        args,
                                        n_workers=self.n_workers,
                                        initializer=self.init_pool,
                                        initargs=(bank_pack,),
                                        msg="unc + simi")

        # with mp.Pool(self.n_workers, initializer=self.init_pool, initargs=(bank_pack,)) as p:
        #     out = []
        #     with tqdm(total=len(args), desc='Processing', ncols=80) as pbar:
        #         for result in p.starmap(self.simi_iter, args):
        #             out.append(result)
        #             pbar.update(1)
        #     out = p.starmap(self.simi_iter, args)

        self.index = [self.buffer_cases[np.argmax(simi_score)]]

    @staticmethod
    def simi_mi_iter(case_n):
        img_indices = [idx for idx, pth in enumerate(shared_pack.bank_paths) if case_n in pth]
        buffer_data = shared_pack.bank_data[img_indices]
        tmp_simi_mat = cosine_similarity(buffer_data, shared_pack.bank_data)
        tmp_simi_mat[tmp_simi_mat < 0] = 0
        tmp_mi_mat = np.zeros((buffer_data.shape[0], shared_pack.train_data.shape[0]), dtype=np.float16)
        for idx in range(buffer_data.shape[0]):
            for j in range(shared_pack.train_data.shape[0]):
                tmp_mi_mat[idx, j] = VolumeCluster.mutual_info(buffer_data[idx], shared_pack.train_data[j])

        return np.mean(np.sum(tmp_simi_mat, axis=1)), np.mean(np.sum(tmp_mi_mat, axis=1))

    def simi_mi(self, args):
        mix_pack = namedtuple("DataPack", ["bank_paths", "bank_data", "train_data"])(self.bank_paths,
                                                                                     self.bank_data,
                                                                                     self.train_data)

        # with mp.Pool(self.n_workers, initializer=self.init_pool, initargs=(mix_pack,)) as p:
        #     out = p.starmap(self.simi_mi_iter, args)
        out = multiprocess_agent(self.simi_mi_iter,
                                 args,
                                 n_workers=self.n_workers,
                                 initializer=self.init_pool,
                                 initargs=(mix_pack,),
                                 msg="unc + simi + mi")

        print("Number of candidate cases: {}.".format(len(out)))
        simi_score, mi_score = zip(*out)
        score = self.min_max(np.array(simi_score)) - self.coef * self.min_max(np.array(mi_score))
        self.index = [self.buffer_cases[np.argmax(score)]]

    @staticmethod
    def mi_iter(case_n):
        img_indices = [idx for idx, pth in enumerate(shared_pack.bank_paths) if case_n in pth]
        buffer_data = shared_pack.bank_data[img_indices]
        tmp_mi_mat = np.zeros((buffer_data.shape[0], shared_pack.train_data.shape[0]), dtype=np.float16)
        for idx in range(buffer_data.shape[0]):
            for j in range(shared_pack.train_data.shape[0]):
                tmp_mi_mat[idx, j] = VolumeCluster.mutual_info(buffer_data[idx], shared_pack.train_data[j])

        return np.mean(np.sum(tmp_mi_mat, axis=1))

    def mi(self, args):
        mix_pack = namedtuple("DataPack", ["bank_paths", "bank_data", "train_data"])(self.bank_paths,
                                                                                     self.bank_data,
                                                                                     self.train_data)

        mi_score = multiprocess_agent(self.mi_iter,
                                      args,
                                      n_workers=self.n_workers,
                                      initializer=self.init_pool,
                                      initargs=(mix_pack,),
                                      msg="unc + mi")
        # with mp.Pool(self.n_workers, initializer=self.init_pool, initargs=(mix_pack,)) as p:
        #     out = []
        #     with tqdm(total=len(args), desc='Processing', ncols=80) as pbar:
        #         for result in p.starmap(self.simi_mi_iter, args):
        #             out.append(result)
        #             pbar.update(1)
        # out = p.starmap(self.simi_mi_iter, args)

        print("Number of candidate cases: {}.".format(len(mi_score)))
        score = self.min_max(np.asarray(self.uncertainty)) - self.min_max(np.asarray(mi_score))
        self.index = [self.buffer_cases[np.argmax(score)]]

    def run(self):
        if self.method == "unc":
            self.index = [self.buffer_cases[0]]
        else:
            self.read_vector()
            args = [(case_n,) for case_n in self.buffer_cases]
            if self.method == "simi":
                self.simi(args)
            elif self.method[:7] == "simi_mi":
                self.simi_mi(args)
            elif self.method == "mi":
                self.mi(args)
            else:
                raise NotImplementedError(f"Method {self.method} is not implemented.")

        return self.index
