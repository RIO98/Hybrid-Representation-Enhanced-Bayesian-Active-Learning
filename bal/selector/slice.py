from abc import ABC
from collections import namedtuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bal.utils.utils import multiprocess_agent
from .base_active_selector import BaseActiveSelector

shared_pack = None


class RandomSelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list):
        super(RandomSelector, self).__init__(file_root,
                                             train_paths,
                                             bank_paths,
                                             n_outputs)

        self.buffer_slices = buffer_slices

    def apply(self):
        self.indices = np.random.choice(self.buffer_slices, self.n_outputs, replace=False)

    def run(self):
        self.apply()

        return self.indices


class UncertaintySelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list):
        super(UncertaintySelector, self).__init__(file_root,
                                                  train_paths,
                                                  bank_paths,
                                                  n_outputs)

        self.buffer_slices = buffer_slices

    def apply(self):
        self.indices = [self.buffer_slices[i] for i in range(self.n_outputs)]

    def run(self):
        self.apply()

        return self.indices


class DensitySelector(BaseActiveSelector, ABC):

    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list,
                 clip=False, img_vmin=0, img_vmax=1000):
        super(DensitySelector, self).__init__(file_root,
                                              train_paths,
                                              bank_paths,
                                              n_outputs,
                                              clip,
                                              img_vmin,
                                              img_vmax)

        self.buffer_slices = buffer_slices

    @staticmethod
    def core_iter(i, indices):
        tmp = indices + [i]  # build a new list to avoid changing self.indices
        tmp_simi_mat = shared_pack.simi_mat[tmp]
        tmp_simi_mat[tmp_simi_mat < 0] = 0
        return np.sum(np.amax(tmp_simi_mat, axis=0))

    def apply(self):
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
            simi_score = multiprocess_agent(self.core_iter,
                                            args,
                                            n_workers=self.n_workers,
                                            initializer=self.init_pool,
                                            initargs=(mat_pack,),
                                            msg="unc + simi")

            self.indices.append(np.argmax(simi_score))
            print(len(self.indices), self.indices[-1])

        self.indices = [x for x in self.indices if x is not None]
        print(len(self.indices))
        if len(self.indices) != self.n_outputs:
            print("Number of indices is inconsistent with n_outputs")

    def run(self):
        self.read_vector()
        self.apply()

        return self.indices


class DiversitySelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list,
                 clip=False, img_vmin=0, img_vmax=1000, uncertainty=None):
        super(DiversitySelector, self).__init__(file_root, train_paths, bank_paths, n_outputs, clip, img_vmin, img_vmax)

        self.buffer_slices = buffer_slices
        self.uncertainty = uncertainty

    @staticmethod
    def core_iter(idx):
        mi = np.zeros((shared_pack.train_data.shape[0]), dtype=np.float16)
        for i in range(shared_pack.train_data.shape[0]):
            mi[i] = DiversitySelector.mutual_info(shared_pack.buffer_data[idx], shared_pack.train_data[i])

        return mi

    def apply(self):
        img_indices = [i for i, pth in enumerate(self.bank_paths) if pth in self.buffer_slices]
        assert len(img_indices) == len(self.buffer_slices), "Buffer slices not found in unlabeled pool!"
        buffer_data = self.bank_data[img_indices]

        args = [(i,) for i in range(len(self.buffer_slices))]
        mix_pack = namedtuple("DataPack", ["buffer_data", "train_data"])(buffer_data, self.train_data)
        out = multiprocess_agent(self.core_iter,
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
        self.read_vector()
        self.apply()

        return self.indices


class HybridSelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_slices: list, n_outputs: int, train_paths: list, bank_paths: list,
                 coef: float = 0.5, clip=False, img_vmin=0, img_vmax=1000):
        super(HybridSelector, self).__init__(file_root, train_paths, bank_paths, n_outputs, clip, img_vmin, img_vmax)

        self.buffer_slices = buffer_slices
        self.coef = coef

    @staticmethod
    def core_iter(i, indices):
        selected_indices = indices + [i]
        tmp_simi_mat = shared_pack.simi_mat[selected_indices]
        tmp_mi_mat = shared_pack.mi_mat[selected_indices]
        tmp_simi_mat[tmp_simi_mat < 0] = 0
        return np.sum(np.amax(tmp_simi_mat, axis=0)), np.sum(np.amax(tmp_mi_mat, axis=0))

    def apply(self):
        img_indices = [i for i, pth in enumerate(self.bank_paths) if pth in self.buffer_slices]
        assert len(img_indices) == len(self.buffer_slices), "Buffer slices not found in unlabeled pool!"
        buffer_data = self.bank_data[img_indices]

        args = [(i,) for i in range(len(self.buffer_slices))]
        mix_pack = namedtuple("DataPack", ["buffer_data", "bank_data"])(buffer_data, self.bank_data)
        out = multiprocess_agent(DiversitySelector.core_iter,
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
            out = multiprocess_agent(self.core_iter,
                                     args,
                                     n_workers=self.n_workers,
                                     initializer=self.init_pool,
                                     initargs=(mat_pack,),
                                     msg="unc + simi + mi")

            simi_score, mi_score = zip(*out)
            score = self.min_max(np.array(simi_score)) - self.coef * self.min_max(np.array(mi_score))
            self.indices.append(np.argmax(score))
            print(max(HybridSelector.min_max(simi_score)), min(HybridSelector.min_max(simi_score)),
                  max(HybridSelector.min_max(mi_score)))
            print(len(self.indices), self.indices[-1])

        self.indices = [x for x in self.indices if x is not None]
        print(len(self.indices))
        if len(self.indices) != self.n_outputs:
            print("Number of indices is inconsistent with n_outputs")

    def run(self):
        self.read_vector()
        self.apply()

        return self.indices
