from abc import ABC
from collections import namedtuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bal.utils.utils import multiprocess_agent
from .base_active_selector import BaseActiveSelector


class RandomSelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list):
        super(RandomSelector, self).__init__(file_root,
                                             train_paths,
                                             bank_paths,
                                             n_outputs)

        self.buffer_cases = buffer_cases

    @staticmethod
    def core_iter(case_name):
        pass

    def apply(self):
        self.indices = np.random.choice(self.buffer_cases, self.n_outputs, replace=False)

    def run(self):
        self.apply()

        return self.indices


class UncertaintySelector(BaseActiveSelector, ABC):
    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list):
        super(UncertaintySelector, self).__init__(file_root,
                                                  train_paths,
                                                  bank_paths,
                                                  n_outputs)

        self.buffer_cases = buffer_cases

    @staticmethod
    def core_iter(case_name):
        pass

    def apply(self):
        self.indices = [self.buffer_cases[i] for i in range(self.n_outputs)]

    def run(self):
        self.apply()

        return self.indices


class DensitySelector(BaseActiveSelector, ABC):

    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list,
                 clip=False, img_vmin=0, img_vmax=1000):
        super(DensitySelector, self).__init__(file_root,
                                              train_paths,
                                              bank_paths,
                                              n_outputs,
                                              clip,
                                              img_vmin,
                                              img_vmax)

        self.buffer_cases = buffer_cases

    @staticmethod
    def init_dpack(data_pack):
        global dpack
        dpack = data_pack

    @staticmethod
    def core_iter(case_name):
        img_indices = [i for i, pth in enumerate(dpack.bank_paths) if case_name in pth]
        buffer_data = dpack.bank_data[img_indices]
        tmp_simi_mat = cosine_similarity(buffer_data, dpack.bank_data)
        tmp_simi_mat[tmp_simi_mat < 0.] = 0.
        return np.sum(np.amax(tmp_simi_mat, axis=0))

    def apply(self):
        args = [(case_n,) for case_n in self.buffer_cases]
        bank_pack = namedtuple("DataPack", ["bank_paths", "bank_data"])(self.bank_paths, self.bank_data)

        simi_score = multiprocess_agent(self.core_iter,
                                        args,
                                        n_workers=self.n_workers,
                                        initializer=self.init_dpack,
                                        initargs=(bank_pack,),
                                        msg="unc + simi")

        print(f"Number of candidate cases: {len(simi_score)}.")
        self.indices = [self.buffer_cases[np.argmax(simi_score)]]

    def run(self):
        self.read_vector()
        self.apply()

        return self.indices


class DiversitySelector(BaseActiveSelector, ABC):

    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list,
                 clip=False, img_vmin=0, img_vmax=1000, uncertainty=None):
        super(DiversitySelector, self).__init__(file_root,
                                                train_paths,
                                                bank_paths,
                                                n_outputs,
                                                clip,
                                                img_vmin,
                                                img_vmax)

        self.buffer_cases = buffer_cases
        self.uncertainty = uncertainty

    @staticmethod
    def init_dpack(data_pack):
        global dpack
        dpack = data_pack

    @staticmethod
    def core_iter(idx):
        # Multiprocess on the number of buffer cases
        # img_indices = [idx for idx, pth in enumerate(dpack.bank_paths) if case_n in pth]
        # buffer_data = dpack.bank_data[img_indices]
        # tmp_mi_mat = np.zeros((buffer_data.shape[0], dpack.train_data.shape[0]), dtype=np.float16)
        # for idx in range(buffer_data.shape[0]):
        #     for j in range(dpack.train_data.shape[0]):
        #         tmp_mi_mat[idx, j] = DiversitySelector.mutual_info(buffer_data[idx], dpack.train_data[j])
        #
        # return np.mean(np.sum(tmp_mi_mat, axis=1))

        # Multiprocess on the number of slices for each buffer case
        mi_vec = np.zeros((dpack.train_data.shape[0],), dtype=np.float16)
        for i in range(dpack.train_data.shape[0]):
            mi_vec[i] = DiversitySelector.mutual_info(dpack.buffer_data[idx], dpack.train_data[i])

        return mi_vec

    def apply(self):
        mi_score = []
        for case_name in self.buffer_cases:
            img_indices = [i for i, pth in enumerate(self.bank_paths) if case_name in pth]
            buffer_data = self.bank_data[img_indices]
            args = [(i,) for i in range(buffer_data.shape[0])]
            mix_pack = namedtuple("DataPack", ["buffer_data", "train_data"])(buffer_data, self.train_data)

            mi_vectors = multiprocess_agent(self.core_iter,
                                            args,
                                            n_workers=self.n_workers,
                                            initializer=self.init_dpack,
                                            initargs=(mix_pack,),
                                            msg="unc + mi")
            mi_score.append(np.mean(np.vstack(mi_vectors)))

        print(f"Number of candidate cases: {len(mi_score)}.")
        score = self.min_max(np.asarray(self.uncertainty)) - self.min_max(np.asarray(mi_score))
        self.indices = [self.buffer_cases[np.argmax(score)]]

    def run(self):
        self.read_vector()
        self.apply()

        return self.indices


class HybridSelector(BaseActiveSelector, ABC):

    def __init__(self, file_root: str, buffer_cases: list, n_outputs: int, train_paths: list, bank_paths: list,
                 coef: float = 0.5, clip=False, img_vmin=0, img_vmax=1000):
        super(HybridSelector, self).__init__(file_root,
                                             train_paths,
                                             bank_paths,
                                             n_outputs,
                                             clip,
                                             img_vmin,
                                             img_vmax)

        self.buffer_cases = buffer_cases
        self.coef = coef

    @staticmethod
    def init_dpack(data_pack):
        global dpack
        dpack = data_pack

    @staticmethod
    def core_iter(case_n):
        print("Implement this when the number of buffer cases is large or when there are only a few slices in a "
              "volume image.")
        # Multiprocess on the number of buffer cases
        # img_indices = [i for i, pth in enumerate(dpack.bank_paths) if case_n in pth]
        # buffer_data = dpack.bank_data[img_indices]
        # tmp_simi_mat = cosine_similarity(buffer_data, dpack.bank_data)
        # tmp_simi_mat[tmp_simi_mat < 0] = 0
        # tmp_mi_mat = np.zeros((buffer_data.shape[0], dpack.train_data.shape[0]), dtype=np.float16)
        # for i in range(buffer_data.shape[0]):
        #     for j in range(dpack.train_data.shape[0]):
        #         tmp_mi_mat[i, j] = HybridSelector.mutual_info(buffer_data[i], dpack.train_data[j])
        #
        # return np.mean(np.sum(tmp_simi_mat, axis=1)), np.mean(np.sum(tmp_mi_mat, axis=1))

    def apply(self):
        mi_score, simi_score = [], []
        for case_name in self.buffer_cases:
            img_indices = [i for i, pth in enumerate(self.bank_paths) if case_name in pth]
            buffer_data = self.bank_data[img_indices]
            args = [(i,) for i in range(buffer_data.shape[0])]
            mix_pack = namedtuple("DataPack", ["buffer_data", "train_data"])(buffer_data,
                                                                             self.train_data)
            mi_mat = multiprocess_agent(DiversitySelector.core_iter,
                                        args,
                                        n_workers=self.n_workers,
                                        initializer=self.init_dpack,
                                        initargs=(mix_pack,),
                                        msg="Get MI score")
            mi_score.append(np.sum(np.amax(np.vstack(mi_mat), axis=0)))

            simi_mat = cosine_similarity(buffer_data, self.bank_data)
            simi_mat[simi_mat < 0] = 0

            simi_score.append(np.sum(np.amax(simi_mat, axis=0)))

        assert len(simi_score) == len(mi_score), \
            f"The length of similarity score and MI score are not equal: {len(simi_score)}:{len(mi_score)}"
        print(f"Number of candidate cases: {len(self.buffer_cases)}.")
        score = self.min_max(np.array(simi_score)) - self.coef * self.min_max(np.array(mi_score))
        self.indices = [self.buffer_cases[np.argmax(score)]]

    def run(self):
        self.read_vector()
        self.apply()

        return self.indices
