import os
import random
import urllib.request
import zipfile
from glob import glob
from pathlib import Path

import numpy as np
import pydicom
import tqdm
import cv2

from bal.data.io.image_file_handler import ImageFileHandler
from bal.utils.image_processor import ImageProcess
from bal.utils.utils import multiprocess_agent, write_txt


_SHAPE = (320, 440)

def my_hook(t):  # https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).

    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download(url, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not os.path.exists(out):
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, ncols=80) as t:
            urllib.request.urlretrieve(url, out, reporthook=my_hook(t))


def unzip(zip_file, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with zipfile.ZipFile(zip_file) as existing_zip:
        existing_zip.extractall(out)


def dcm2mha(f, data_type, out_dir):
    dcm = pydicom.dcmread(f)
    element_spacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SpacingBetweenSlices))
    array_data = np.transpose(dcm.pixel_array, (1, 2, 0))  # (z, y, x) -> (y, x, z)

    if data_type == "label":
        inter_method = cv2.INTER_NEAREST
    elif data_type == "image":
        inter_method = cv2.INTER_CUBIC
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")

    case_name = os.path.basename(os.path.splitext(f)[0])
    save_dir = os.path.join(out_dir, case_name)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(array_data.shape[2]):
        f_name = f"{data_type}_{i:04d}.mha"
        padded_image = ImageProcess.pad_to_target_shape(array_data[:, :, i], _SHAPE)
        resized_image = ImageProcess.half_size_image(padded_image, inter_method)
        ImageFileHandler.save_image(os.path.join(save_dir, f_name), np.expand_dims(resized_image, axis=-1), element_spacing)


def preprocess_images(image_dir, out_dir):
    image_files = glob(os.path.join(image_dir, "data", "*.dcm"))
    print(f"# test images: {len(image_files)}")

    _ = multiprocess_agent(dcm2mha, [(f, "image", out_dir) for f in image_files], show_progress=True)

    label_files = glob(os.path.join(image_dir, "groundtruth", "*.dcm"))
    print(f"# test images: {len(label_files)}")

    _ = multiprocess_agent(dcm2mha, [(f, "label", out_dir) for f in label_files], show_progress=True)


def generate_slices(case_list, data_root):
    slices = []
    for case in case_list:
        slices += [str(Path(r).parent.relative_to(data_root) / Path(r).name) for r in
                   glob(os.path.join(data_root, case, "image*.mha"))]
    return slices


def split_data(data_root, txt_dir, train=0.7, valid=0.15):
    seed = 0
    exps = ["volume", "slice"]
    methods = ["random", "unc", "mi", "simi", "simi_mi"]
    random.seed(seed)

    assert train + valid < 1., "Ratio sum of train and validation should be less than 1.0"
    patients = [os.path.basename(r) for r in glob(os.path.join(data_root, "*"))]
    random.shuffle(patients)
    t, v = int(len(patients) * train), int(len(patients) * (train + valid))
    train_cases, bank_cases, val_cases, test_cases = patients[:1], patients[1:t], patients[t:v], patients[v:]

    print(
        f"number of testing cases: {len(test_cases)}, number of validation cases: {len(val_cases)}, number of unlabeled cases: {len(bank_cases)}")

    case_lists = {"training": train_cases, "databank": bank_cases, "testing": test_cases, "validation": val_cases}
    slices = {k: generate_slices(v, data_root) for k, v in case_lists.items()}

    for exp_n in exps:
        for meth in methods:
            save_root = os.path.join(txt_dir, exp_n, f"seed_{seed}", "4layer", meth)
            os.makedirs(save_root, exist_ok=True)
            print(f"save in {save_root}")
            for k, v in slices.items():
                write_txt(v, os.path.join(save_root, f"id-list_trial-1_{k}-0.txt"))


def preprocess_data(out_dir, temp_dir=None):
    if temp_dir is None:
        temp_dir = os.path.join(out_dir, "temp")

    data_url = "https://opencas.webarchiv.kit.edu/data/thyroid.zip"
    data_zip = os.path.join(temp_dir, os.path.basename(data_url))
    image_dir = os.path.join(temp_dir, "image")

    download(data_url, data_zip)
    unzip(data_zip, image_dir)

    save_dir = os.path.join(out_dir, "us_thyroid")
    os.makedirs(save_dir, exist_ok=True)
    preprocess_images(image_dir, save_dir)
    split_data(save_dir, os.path.join(out_dir, "experiments"))


if __name__ == '__main__':
    out_dir = "./preprocessed"

    preprocess_data(out_dir)
