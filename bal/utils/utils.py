# Code adapted from https://github.com/yuta-hi/pytorch_bayesian_unet/blob/master/pytorch_bcnn/utils/__init__.py

import argparse
import contextlib
import inspect
import json
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import sys
import shutil
import tempfile
import warnings

import numpy as np
from tqdm import tqdm


@contextlib.contextmanager
def fixed_seed(seed, strict=False):
    """Fix random seed to improve the reproducibility.

    Args:
        seed (float): Random seed
        strict (bool, optional): If True, cuDNN works under deterministic mode.
            Defaults to False.

    TODO: Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`.
          If your dataset has stochastic behavior, such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.
    """

    import random
    import torch
    import copy

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if strict:
        warnings.warn('Even if `strict` is set to True, the reproducibility cannot be guaranteed under the `MultiprocessIterator`. \
          If your dataset has stochastic behavior such as data augmentation, you should use the `SerialIterator` or `MultithreadIterator`.')

        _deterministic = copy.copy(torch.backends.cudnn.deterministic)
        _benchmark = copy.copy(torch.backends.cudnn.benchmark)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yield

    if strict:
        torch.backends.cudnn.deterministic = _deterministic
        torch.backends.cudnn.benchmark = _benchmark


# https://github.com/chainer/chainerui/blob/master/chainerui/utils/tempdir.py
@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)


# https://github.com/chainer/chainerui/blob/master/chainerui/utils/save_args.py
def convert_dict(conditions):
    if isinstance(conditions, argparse.Namespace):
        return vars(conditions)
    return conditions


# https://github.com/chainer/chainerui/blob/master/chainerui/utils/save_args.py
def save_args(conditions, out_path):
    """A util function to save experiment condition for job table.

    Args:
        conditions (:class:`argparse.Namespace` or dict): Experiment conditions
            to show on a job table. Keys are show as table header and values
            are show at a job row.
        out_path (str): Output directory name to save conditions.

    """

    args = convert_dict(conditions)

    try:
        os.makedirs(out_path)
    except OSError:
        pass

    with tempdir(prefix='args', dir=out_path) as tempd:
        path = os.path.join(tempd, 'args.json')
        with open(path, 'w') as f:
            json.dump(args, f, indent=4)

        new_path = os.path.join(out_path, 'args')
        shutil.move(path, new_path)


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    return lines


def write_txt(content, pth):
    with open(pth, 'w') as f:
        for item in content:
            f.write("%s\n" % item)


def multiprocess_agent(func, args_list, n_workers=mp.cpu_count() - 1, initializer=None, initargs=(), msg='Processing',
                       show_progress=False):
    if not isinstance(initargs, (list, tuple)):
        initargs = tuple([initargs])

    minor_version = sys.version.split('.')[1]

    with mp.Pool(n_workers, initializer=initializer, initargs=initargs) as p:
        if show_progress and minor_version >= '8':
            out = []
            for result in tqdm(p.istarmap(func, args_list), total=len(args_list),
                               desc=f"{msg}, n_workers: {n_workers}"):
                out.append(result)
        else:
            out = p.starmap(func, args_list)

    return out


def get_init_parameter_names(cls):
    init_signature = inspect.signature(cls.__init__)
    parameter_names = [
        param.name for param in init_signature.parameters.values() if param.name != "self"
    ]
    return parameter_names


def istarmap(self, func, iterable, chunksize=1):
    """
    https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm, Darkonaut
    starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
