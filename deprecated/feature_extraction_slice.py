import multiprocessing as mp
import os
from collections import namedtuple

import numpy as np
import psutil
from sklearn.metrics.pairwise import cosine_similarity

from bal.data.io.image_file_handler import ImageFileHandler
from bal.utils.image_processor import ImageProcess
from bal.utils import multiprocess_agent


class SliceCluster:
    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list,
                 method: str = 'simi', coef: float = 0.5, clip=False, img_vmin=0, img_vmax=1000,
                 uncertainty: list = None):
        self.file_root = file_root
        self.method = method
        self.uncertainty = uncertainty
        self.train_paths = train_paths
        self.bank_paths = bank_paths
        self.buffer_slices = buffer_slices
        self.n_outputs = n_outputs
        self.indices = []
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
            # args = []
            # for i in range(len(self.train_paths)):
            #     args.append([i])
            #
            # with mp.Pool(self.cpu_count, initializer=SliceCluster.init_pool5,
            #              initargs=(self.file_root, self.train_paths, self.clip, self.ww, self.wl)) as p:
            #     out = p.starmap(SliceCluster.vec_reader, args)
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
            # args = []
            # for i in range(len(self.bank_paths)):
            #     args.append([i])
            #
            # with mp.Pool(self.cpu_count, initializer=SliceCluster.init_pool5,
            #              initargs=(self.file_root, self.bank_paths, self.clip, self.ww, self.wl)) as p:
            #     out = p.starmap(SliceCluster.vec_reader, args)
            out = multiprocess_agent(self.vec_reader,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(bank_data_pack,),
                                     msg="Load unlabeled data")

            self.bank_data = np.vstack(out)

        print("Data loaded!")

    @staticmethod
    def simi_iter(i, indices):
        tmp = indices + [i]  # build a new list to avoid changing self.indices
        tmp_simi_mat = shared_pack.simi_mat[tmp]
        tmp_simi_mat[tmp_simi_mat < 0] = 0
        return np.sum(np.amax(tmp_simi_mat, axis=0))

    def simi(self):
        # Calculate similarity matrix
        img_indices = [i for i, pth in enumerate(self.bank_paths) if pth in self.buffer_slices]
        assert len(img_indices) == len(self.buffer_slices), "Buffer slices not found in unlabeled pool!"
        buffer_data = self.bank_data[img_indices]
        simi_mat = cosine_similarity(buffer_data, self.bank_data)
        simi_mat[simi_mat < 0] = 0

        # [i, self.indices] needs to be mutable for dynamic update
        args = [[i, self.indices] for i in range(len(self.buffer_slices))]
        mat_pack = namedtuple("DataPack", ["simi_mat"])(simi_mat)

        while len(self.indices) < self.n_outputs:
            simi_score = multiprocess_agent(self.simi_iter,
                                            args,
                                            n_workers=self.n_workers,
                                            initializer=self.init_pool,
                                            initargs=(mat_pack,),
                                            msg="unc + simi")
            # with mp.Pool(self.cpu_count, initializer=self.init_pool, initargs=(simi_mat,)) as p:
            #     out = p.starmap(SliceCluster.simi_iter, args)

            self.indices.append(np.argmax(simi_score))
            print(len(self.indices), self.indices[-1])

        self.indices = [x for x in self.indices if x is not None]
        print(len(self.indices))
        if len(self.indices) != self.n_outputs:
            print("Number of indices is inconsistent with n_outputs")

    @staticmethod
    def simi_mi_iter(i, indices):
        selected_indices = indices + [i]
        tmp_simi_mat = shared_pack.simi_mat[selected_indices]
        tmp_mi_mat = shared_pack.mi_mat[selected_indices]
        tmp_simi_mat[tmp_simi_mat < 0] = 0
        return np.sum(np.amax(tmp_simi_mat, axis=0)), np.sum(np.amax(tmp_mi_mat, axis=0))

    def simi_mi(self):
        img_indices = [i for i, pth in enumerate(self.bank_paths) if pth in self.buffer_slices]
        assert len(img_indices) == len(self.buffer_slices), "Buffer slices not found in unlabeled pool!"
        buffer_data = self.bank_data[img_indices]

        args = [(i,) for i in range(len(self.buffer_slices))]
        mix_pack = namedtuple("DataPack", ["buffer_data", "bank_data"])(buffer_data, self.bank_data)

        # with mp.Pool(self.cpu_count, initializer=SliceCluster.init_pool2, initargs=(self.buffer_data, self.train_data,)) as p:
        #     out = p.starmap(SliceCluster.mi_iter, args)
        out = multiprocess_agent(self.mi_iter,
                                 args,
                                 n_workers=self.n_workers,
                                 initializer=self.init_pool,
                                 initargs=(mix_pack,),
                                 msg="Calculate MI matrix")

        mi_mat = np.vstack(out)
        simi_mat = cosine_similarity(buffer_data, self.bank_data)
        print(f"MI matrix shape: {mi_mat.shape}; similarity matrix shape: {simi_mat.shape}")

        # [i, self.indices] needs to be mutable for dynamic update
        args = [[i, self.indices] for i in range(len(self.buffer_slices))]
        mat_pack = namedtuple("DataPack", ["simi_mat", "mi_mat"])(simi_mat, mi_mat)

        while len(self.indices) < self.n_outputs:
            # with mp.Pool(self.cpu_count, initializer=SliceCluster.init_pool3,
            #              initargs=(simi_mat, mi_mat, self.indices)) as p:
            #     out = p.starmap(SliceCluster.simi_mi_iter, args)
            out = multiprocess_agent(self.simi_mi_iter,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(mat_pack,),
                                     msg="unc + simi + mi")

            simi_score, mi_score = zip(*out)
            score = self.min_max(np.array(simi_score)) - self.coef * self.min_max(np.array(mi_score))
            self.indices.append(np.argmax(score))
            print(max(SliceCluster.min_max(simi_score)), min(SliceCluster.min_max(simi_score)),
                  max(SliceCluster.min_max(mi_score)))
            print(len(self.indices), self.indices[-1])

        self.indices = [x for x in self.indices if x is not None]
        print(len(self.indices))
        if len(self.indices) != self.n_outputs:
            print("Number of indices is inconsistent with n_outputs")

    @staticmethod
    def mi_iter(idx):
        if idx % 1000 == 0:
            print(f'RAM memory {psutil.virtual_memory()[2]} used:')
        mi = np.zeros((shared_pack.bank_data.shape[0]), dtype=np.float16)
        for i in range(shared_pack.bank_data.shape[0]):
            mi[i] = SliceCluster.mutual_info(shared_pack.buffer_data[idx], shared_pack.bank_data[i])

        return mi

    def mi(self):
        img_indices = [i for i, pth in enumerate(self.bank_paths) if pth in self.buffer_slices]
        assert len(img_indices) == len(self.buffer_slices), "Buffer slices not found in unlabeled pool!"
        buffer_data = self.bank_data[img_indices]

        args = [(i,) for i in range(len(self.buffer_slices))]
        mix_pack = namedtuple("DataPack", ["buffer_data", "bank_data"])(buffer_data, self.bank_data)

        # with mp.Pool(self.cpu_count, initializer=SliceCluster.init_pool2, initargs=(self.buffer_data, self.train_data,)) as p:
        #     out = p.starmap(SliceCluster.mi_iter, args)
        out = multiprocess_agent(self.mi_iter,
                                 args,
                                 n_workers=self.n_workers,
                                 initializer=self.init_pool,
                                 initargs=(mix_pack,),
                                 msg="Calculate MI matrix")

        mi_vec = np.mean(np.vstack(out), axis=1)
        print(mi_vec.shape)

        norm_mi = self.min_max(mi_vec)
        norm_uncert = self.min_max(np.asarray(self.uncertainty))

        assert len(norm_mi) == len(norm_uncert), "number of mi is not consistent with number of uncertainty"

        scores = norm_uncert - norm_mi
        self.indices = np.argpartition(-scores, self.n_outputs)[:self.n_outputs]
        if len(self.indices) != self.n_outputs:
            print("number of indices is inconsistent with n_outputs")

    def run(self):
        if self.method == "unc":
            self.indices = list(range(self.n_outputs))

        else:
            self.read_vector()
            if self.method == "mi":
                self.mi()

            elif self.method == "simi":
                self.simi()

            elif self.method[:7] == "simi_mi":
                self.simi_mi()

        return self.indices
