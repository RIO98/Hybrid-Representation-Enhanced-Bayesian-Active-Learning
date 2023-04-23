import os
from functools import partial

import numpy as np
import torch
import tqdm

from bal.data.io.meta_image import MetaImage
from bal.models.links import MCSampler
from bal.utils.image_processor import ImageProcess


class Inferencer:
    """
    Inference class for running model predictions on given image paths.

    :param root: Root directory for the images.
    :param save_dir: Path to save the output images.
    :param image_paths: List of paths to the input images.
    :param model: Model to be used for inference.
    :param snapshot: Snapshot of the model's state dictionary.
    :param gpu_id: GPU id for running the model.
    :param mc_iteration: Number of Monte Carlo iterations.
    :param clip_range: Range for clipping pixel values.
    :param is_skin: Boolean indicating whether the output is skin mask.
    """

    def __init__(self,
                 save_dir,
                 image_paths,
                 model,
                 snapshot,
                 gpu_id=0,
                 mc_iteration=10,
                 clip_range=(0, 1000),
                 is_skin=False):

        self.save_dir = save_dir
        self.image_paths = image_paths
        self.model = model
        self.snapshot = snapshot
        self.gpu_id = gpu_id
        self.mc_iteration = mc_iteration
        self.clip_range = clip_range
        self.is_skin = is_skin

    def run(self):
        model = MCSampler(self.model,
                          mc_iteration=self.mc_iteration,
                          activation=partial(torch.softmax, dim=1),
                          reduce_mean=partial(torch.argmax, dim=1),
                          reduce_var=partial(torch.mean, dim=1))
        model.load_state_dict(torch.load(self.snapshot))
        print(f"Loaded a snapshot model: {self.snapshot}")

        device = torch.device(self.gpu_id)
        model.to(device)
        model.eval()

        for image_path in tqdm.tqdm(self.image_paths):
            self._predict(image_path, model)

    def _predict(self, volume_file, model):
        spliter = os.path.sep

        patient = volume_file.split(spliter)[-2]  # NOTE: Ad-hoc
        _, file_ext = os.path.splitext(volume_file)
        filename = volume_file.split(spliter)[-1].replace(file_ext, "")  # NOTE: Ad-hoc
        volume, h = MetaImage.read(volume_file)
        assert len(volume.shape) == 3, "Get 2D volume!"
        print(f"# patient ID: {patient}")
        print(f"-- number of slices: {len(volume)}")

        device = torch.device(self.gpu_id)

        with torch.no_grad():
            labels, uncertainty = [], []
            for idx in range(len(volume)):
                image = volume[idx]
                image = ImageProcess.clip_and_norm(image, *self.clip_range)
                image_tensor = torch.from_numpy(image[np.newaxis, np.newaxis]).type(torch.FloatTensor)
                image_tensor = image_tensor.to(device)
                label, uncert = model(image_tensor)
                labels.append(np.squeeze(label.cpu().numpy()))
                uncertainty.append(np.squeeze(uncert.cpu().numpy()))

        labels = np.asarray(labels).astype(np.int16)
        uncertainty = np.asarray(uncertainty).astype(np.float32)

        pred_save_dir = os.path.join(self.save_dir, patient)
        os.makedirs(pred_save_dir, exist_ok=True)
        print("Save predicted label")
        h["CompressedData"] = True
        MetaImage.write(os.path.join(pred_save_dir, f"{filename}_pred{file_ext}"), labels, h)

        if not self.is_skin:
            print("Save uncertainty")
            MetaImage.write(os.path.join(pred_save_dir, f"{filename}_uncert{file_ext}"), uncertainty, h)
