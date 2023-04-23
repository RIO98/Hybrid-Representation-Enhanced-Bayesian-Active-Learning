import os
import cv2
import numpy as np

from bal.data.io.image_file_handler import FileHandler
import multiprocessing as mp
from glob import glob


def mha2png(pt, file_root, save_root):
    img_files = glob(os.path.join(file_root, pt, 'image_*.mha'))
    lbl_files = glob(os.path.join(file_root, pt, 'muscle_label_*.mha'))
    for img, lbl in zip(img_files, lbl_files):
        img_name = os.path.basename(img).replace('.mha', '.png')
        lbl_name = os.path.basename(lbl).replace('.mha', '.png')
        img_save = os.path.join(save_root, pt, img_name)
        lbl_save = os.path.join(save_root, pt, lbl_name)
        os.makedirs(os.path.dirname(img_save), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_save), exist_ok=True)
        img_f, _ = FileHandler.load_image(img)
        lbl_f, _ = FileHandler.load_image(lbl)
        img_f = np.clip(img_f, 0, 1000)
        img_f = (img_f - 0) / (1000 - 0) * 255
        cv2.imwrite(img_save, img_f.astype(np.uint8))
        cv2.imwrite(lbl_save, lbl_f)


if __name__ == '__main__':
    args = []
    root = r'D:\Database\dataset\MR_Quad\256'
    save_root = r'D:\Database\mr_png'
    patients = [os.path.basename(r) for r in glob(os.path.join(root, '*'))]

    for pt in patients:
        args.append([pt, root, save_root])

    with mp.Pool(mp.cpu_count() - 1) as p:
        out = p.starmap(mha2png, args)
