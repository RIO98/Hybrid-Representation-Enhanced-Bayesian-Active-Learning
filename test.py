import argparse
import os
from glob import glob

import numpy as np
import torch

from bal.inference.inferencer import Inferencer
from bal.models import BayesianUNet


def get_args():
    parser = argparse.ArgumentParser(description='BCNN UNet segmentation Tester')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--data_root', default='../../dataset')
    # parser.add_argument('--exp_name', default='random', help='Experiment (method) name')
    parser.add_argument('--slice_list', default=' ./exp/1/id-list_trial-1_testing-0.txt')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--mc_iteration', type=int, default=10)
    parser.add_argument('--img_vmin', type=float, default=0.)
    parser.add_argument('--img_vmax', type=float, default=255.)
    parser.add_argument('--model_save_dir', default='./logs')
    parser.add_argument('--iteration', default='final')
    parser.add_argument('--dataset', default='slice-wise')
    parser.add_argument('--pred_save_dir', '-o', default='./result', help='Output directory')
    return parser.parse_args()


def setup_predictor(args):
    """Set up the predictor."""
    conv_param = {
        'name': 'conv',
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'padding_mode': 'reflect',
        'initialW': {'name': 'he_normal'},
        'initial_bias': {'name': 'zero'},
    }

    upconv_param = {
        'name': 'deconv',
        'kernel_size': 3,
        'stride': 2,
        'padding': 0,
        'initialW': {'name': 'bilinear'},
        'initial_bias': {'name': 'zero'},
    }

    norm_param = {'name': 'batch'}
    test_slices = np.loadtxt(args.slice_list, dtype=str)
    data_paths = os.path.join(args.data_root, '*', 'image_*.mha')
    image_paths = [p for p in glob(data_paths) if os.path.join(*(p.split(os.sep)[-2:])) in test_slices]
    is_skin = False
    assert len(image_paths) != 0, 'No image path loaded.'

    dropout_param = {'name': 'mc_dropout', 'p': .5, }

    predictor = BayesianUNet(ndim=2,
                             in_channels=1,
                             out_channels=args.n_classes,
                             nlayer=args.n_layers,
                             nfilter=32,
                             conv_param=conv_param,
                             upconv_param=upconv_param,
                             dropout_param=dropout_param,
                             norm_param=norm_param)

    model_save_path = os.path.join(args.model_save_dir, args.iteration + '_model.pth')  # NOTE:Ad-hoc

    return predictor, image_paths, model_save_path, is_skin


def main():
    args = get_args()

    print(f"GPU: {args.gpu}")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    predictor, image_paths, model_save_path, is_skin = setup_predictor(args)
    os.makedirs(args.pred_save_dir, exist_ok=True)

    inferencer = Inferencer(save_dir=args.pred_save_dir,
                            image_paths=image_paths,
                            model=predictor,
                            snapshot=model_save_path,
                            gpu_id=args.gpu,
                            mc_iteration=args.mc_iteration,
                            clip_range=(args.img_vmin, args.img_vmax),
                            is_skin=is_skin)

    inferencer.run()


if __name__ == '__main__':
    main()
