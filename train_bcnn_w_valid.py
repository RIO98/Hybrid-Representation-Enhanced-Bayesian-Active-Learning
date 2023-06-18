import copy
import os
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchinfo import summary

from bal.data.augmentor import DataAugmentor, Affine2D, RandomErasing2D, Distort2D, Flip2D
from bal.data.normalizer import Normalizer, Clip2D
from bal.dataloader.slice_image_dataset import SliceImageDataset
from bal.models import BayesianUNet
from bal.models.functions.accuracy import dice_score_2d
from bal.models.functions.loss import focal_loss
from bal.models.functions.loss import softmax_cross_entropy
from bal.models.links import Classifier
from bal.utils import presets
from bal.utils.csv_logger import CSVLogger
from bal.utils.utils import fixed_seed
from bal.utils.utils import save_args
from bal.utils.visualizer import Visualizer
from bal.utils.weight_config import get_class_weight


def tensor2array(var):
    if isinstance(var, torch.Tensor):
        return var.detach().cpu().numpy()
    else:
        raise TypeError('Require tensor input.')


def train_phase(predictor, train, valid, args):
    print(f"# classes: {train.n_classes}")
    print("# samples:")
    print(f"-- train: {len(train)}")
    print(f"-- valid: {len(valid)}")

    device = torch.device(args.gpu)

    train_loader = DataLoader(train, args.batch_size, shuffle=args.shuffle, num_workers=args.n_workers)
    valid_loader = DataLoader(valid, args.batch_size, shuffle=False, num_workers=args.n_workers)
    vis_train_data = copy.deepcopy(train)
    vis_val_data = copy.deepcopy(valid)
    class_weight = get_class_weight(args.cls_weights).to(device)
    case_series = args.train_patients.split(os.sep)[-5]

    if args.loss_func == 'ce':
        loss_fun = partial(softmax_cross_entropy, normalize=False, class_weight=class_weight)
    elif args.loss_func == 'focal':
        loss_fun = partial(focal_loss, weight=class_weight, reduction='sum')
        print("Using focal loss.")
    else:
        raise NotImplementedError("Invalid loss function.")
    # criterion = torch.nn.CrossEntropyLoss()

    model = Classifier(predictor, lossfun=loss_fun)
    model.to(device)
    summary(model, input_size=(8, 1, 256, 256))

    # setup optimizers
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=max(1e-5, 0))
    print(f"Optimizer: AdamW, lr: {args.lr}, weight_decay: {max(1e-5, 0)}")

    # Set scheduler
    # scheduler = CosineAnnealingLR(optimizer, args.iteration)

    # Set visualizer
    visualizer = Visualizer()

    # Set logger
    header = ['epoch', 'iteration', 'training/loss', 'time cost']
    csv_path = os.path.join(args.model_save_dir, 'logs.csv')
    logger = CSVLogger(csv_path, header)
    step = args.train_patients.split(os.sep)[-1].replace('id-list_trial-', '').replace('_training-0.txt', '')
    exp_name = f"{args.train_patients.split(os.sep)[-2]}"
    writer = SummaryWriter(f"runs/{case_series}/seed_{args.seed}/{args.n_layers}layer/{exp_name}/stage{step}")
    print(f"{exp_name}_stage_{step}")

    # Initialize variables
    loss_temp = 0
    n_val = 0
    epoch = 1
    n_iter = len(train_loader)
    n_epoch = np.ceil(args.iteration / n_iter).astype(np.int32)
    n_vis = 10

    print(f"Epoch: {epoch}/{n_epoch}")

    train_iter = iter(train_loader)

    model.train()
    start = time.time()

    vis_val_indices = np.random.choice(np.arange(len(vis_val_data) // args.batch_size),
                                       n_vis,
                                       replace=False)

    for itr in range(args.iteration):
        try:
            image, label = next(train_iter)
        except:  # If all data is loaded, back to the first
            epoch += 1
            print(f"Epoch: {epoch}/{n_epoch}")
            train_iter = iter(train_loader)
            image, label = next(train_iter)

        # cpu -> gpu
        image = image.to(device)
        label = label.to(device)

        # Loss backward
        loss = model(image, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # scheduler.step()
        loss_temp += loss.item()

        # Get learning rate
        # current_lr = scheduler.get_lr()
        current_lr = [args.lr]

        # Print to console
        if (itr + 1) % args.disp_freq == 0:
            end = time.time()
            if itr > 0:
                loss_temp /= (args.disp_freq + 1)

            print(
                f"[epoch {epoch:2d}][iter {itr + 1:4d}/{args.iteration:4d}] "
                f"lr: {current_lr[-1]:.6f} training/loss: {loss_temp:.6f}, time cost: {end - start:.6f}")

            logger([epoch, itr + 1, loss_temp, end - start])
            writer.add_scalar('train_loss', loss_temp, itr + 1)
            writer.add_scalar('learning_rate', current_lr[-1], itr + 1)
            loss_temp = 0
            start = time.time()

        # Save the predicted images
        if (itr + 1) % args.show_freq == 0:
            sample_train_images = []
            vis_train_indices = np.random.choice(np.arange(len(vis_train_data)),
                                                 n_vis,
                                                 replace=True)
            with torch.no_grad():
                for idx in vis_train_indices:
                    image, label = vis_train_data[idx]
                    image = image.unsqueeze(0)  # (c, h, w) -> (n, c, h, w)
                    image = image.to(device)
                    label = label.to(device)
                    out = predictor(image)  # NOTE
                    sample_train_images.append({
                        'image': tensor2array(image),  # (n, c, h, w)
                        'label': tensor2array(out),  # (n, c, h, w)
                        'gt': tensor2array(label)})  # (h, w)

            # Save as images
            os.makedirs(os.path.join(args.model_save_dir, 'train_screenshots'), exist_ok=True)
            visualizer(sample_train_images,
                       os.path.join(args.model_save_dir, 'train_screenshots', 'iteration_{}.png'.format(itr + 1)))

        # Validation phase
        if (itr + 1) % args.valid_freq == 0:
            if itr < 1:
                continue
            print("Start validation\n")
            model.eval()
            val_loss_temp = 0
            val_dc_temp = 0
            sample_val_images = []
            with torch.no_grad():
                for idx, images in enumerate(tqdm.tqdm(valid_loader, desc='Validation', ncols=80), 1):
                    image = images[0].to(device)
                    label = images[1].to(device)
                    out = predictor(image)  # NOTE

                    val_loss = model(image, label)
                    val_loss_temp += val_loss.item()

                    array_label = np.squeeze(np.argmax(tensor2array(out), axis=1).astype(int))
                    array_gt = np.squeeze(tensor2array(label).astype(int))

                    for i in range(array_gt.shape[0]):
                        val_dc_temp += dice_score_2d(array_gt[i], array_label[i])

                    if idx in vis_val_indices:
                        sample_val_images.append({
                            'image': tensor2array(image[:1]),
                            'label': tensor2array(out[:1]),
                            'gt': tensor2array(label[0])})

            # Save as image
            os.makedirs(os.path.join(args.model_save_dir, 'val_screenshots'), exist_ok=True)
            visualizer(sample_val_images,
                       os.path.join(args.model_save_dir, 'val_screenshots', 'iteration_{}.png'.format(itr + 1)))

            total_val_loss = val_loss_temp / (
                    idx * args.batch_size)  # average loss, as the model returns the sum of loss values
            total_val_dc = val_dc_temp / (idx * args.batch_size)  # average dc
            if n_val == 0:
                best_val_loss = total_val_loss
                best_val_dc = total_val_dc

            n_val += 1

            print(f"\nvalidation loss: {np.round(total_val_loss, 4)}, validation dc: {np.round(total_val_dc, 4)}")

            if total_val_dc > best_val_dc:
                best_val_dc = total_val_dc
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'final_model.pth'))
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'best_dc_model.pth'))

            if best_val_loss > total_val_loss:
                best_val_loss = total_val_loss
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'best_loss_model.pth'))

            writer.add_scalar('val_loss', total_val_loss, itr + 1)
            writer.add_scalar('val_dc', total_val_dc, itr + 1)
            model.train()

        if (itr + 1) % 5000 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'model_iter{}.pth'.format(itr + 1)))


def get_dataset(data_root,
                image_ext,
                label_ext,
                train_patients,
                valid_patients,
                normalizer=None,
                augmentor=None,
                dataset='case-wise',
                train_aug=False,
                valid_aug=False):
    class_list = presets.class_list
    dtypes = OrderedDict({'image': np.float32, 'label': np.int64})
    exts = OrderedDict({'image': image_ext, 'label': label_ext})

    if dataset == 'slice-wise':
        getter = partial(SliceImageDataset, root=data_root, classes=class_list,
                         dtypes=dtypes, exts=exts, normalizer=normalizer)
        train_filenames = OrderedDict({
            'image': '{root}/{patients}',
            'label': '{root}/{patients}',
        })
    else:
        raise NotImplementedError('Only slice-wise (2D) dataset is supported now.')

    if train_aug:
        train = getter(patients=train_patients, filenames=train_filenames, augmentor=augmentor)
    else:
        train = getter(patients=train_patients, filenames=train_filenames, augmentor=None)

    if valid_aug:
        valid = getter(patients=valid_patients, filenames=train_filenames, augmentor=augmentor)
    else:
        valid = getter(patients=valid_patients, filenames=train_filenames, augmentor=None)

    print(f"Augmentation for training: {train_aug}; for validation: {valid_aug}")

    return train, valid


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='BCNN UNet segmentation trainer')
    parser.add_argument('--T_0', type=int, default=5000)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--data_root', default='./experiments')
    parser.add_argument('--image_ext', default='image_*.mha')
    parser.add_argument('--label_ext', default='label_*.mha')
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--iteration', type=int, default=10000)
    parser.add_argument('--train_patients', default='./id_list_trial-0_training-0.txt')
    parser.add_argument('--valid_patients', default='./id_list_trial-0_validating-0.txt')
    parser.add_argument('--img_vmin', type=float, default=0.)
    parser.add_argument('--img_vmax', type=float, default=255.)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--shuffle', '-s', action='store_false')
    parser.add_argument('--valid_augment', action='store_true')
    parser.add_argument('--train_augment', action='store_true')
    parser.add_argument('--valid_freq', type=int, default=5000)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--show_freq', type=int, default=1000)
    parser.add_argument('--disp_freq', type=int, default=10)
    parser.add_argument('--lr', '-l', type=float, default=4e-4)
    parser.add_argument('--cls_weights', default='binary')
    parser.add_argument('--dataset', type=str, default='slice-wise')
    parser.add_argument('--loss_func', type=str, default='focal',
                        help='Loss function', choices=['ce', 'focal'])
    parser.add_argument('--freeze_upconv', action='store_true',
                        help='Disables updating the up-convolutional weights. If weights are initialized with \
                            bilinear kernels, up-conv acts as bilinear upsampler.')
    parser.add_argument('--model_save_dir', default='./logs', help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Fix the random seed')
    return parser.parse_args()


def main():
    args = get_args()

    print(f"GPU: {args.gpu}")
    print(f"# Minibatch-size: {args.batch_size}")
    print("")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # setup augmentor
    augmentor = DataAugmentor(n_dim=2)
    augmentor.add(Flip2D(probability=0.3, axis=2))
    augmentor.add(Affine2D(probability=0.95,
                           rotation=10.,
                           translate=(.25, .25),
                           shear=np.pi / 8.,
                           fill_mode=('constant', 'nearest'),
                           zoom=(0.75, 1.25),
                           cval=(0., 0.),
                           interp_order=(1, 0)))

    if not args.valid_augment:
        augmentor.add(Distort2D(probability=0.3,
                                alpha=(75, 125),
                                sigma=10,
                                order=(3, 0)))
        augmentor.add(RandomErasing2D(probability=0.7,
                                      size=(0.02, 0.2),
                                      ratio=(1.0, 0.3),
                                      value_range=(0, 1)))

    # setup normalzier
    normalizer = Normalizer(n_dim=2)
    normalizer.add(Clip2D((args.img_vmin, args.img_vmax)))

    with fixed_seed(args.seed, strict=False):
        # setup a predictor
        conv_param = {  # NOTE: you can change a layer type if you want.
            'name': 'conv',
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'padding_mode': 'reflect',
            'initialW': {'name': 'he_normal'},
            'initial_bias': {'name': 'zero'},
        }

        upconv_param = {  # NOTE: you can change a layer type if you want.
            'name': 'deconv',
            'kernel_size': 3,
            'stride': 2,
            'padding': 0,
            'initialW': {'name': 'bilinear'},
            'initial_bias': {'name': 'zero'},
        }

        norm_param = {'name': 'batch'}

        predictor = BayesianUNet(ndim=2,
                                 in_channels=1,
                                 out_channels=args.n_classes,
                                 nlayer=args.n_layers,
                                 nfilter=32,
                                 conv_param=conv_param,
                                 upconv_param=upconv_param,
                                 norm_param=norm_param)

        if args.freeze_upconv:
            predictor.freeze_layers(name='upconv',
                                    recursive=True,
                                    verbose=True)

        train_patients = np.loadtxt(args.train_patients, dtype=str)
        valid_patients = np.loadtxt(args.valid_patients, dtype=str)
        print(f"Slices for training: {train_patients}")
        train, valid = get_dataset(args.data_root,
                                   args.image_ext,
                                   args.label_ext,
                                   train_patients,
                                   valid_patients,
                                   normalizer,
                                   augmentor,
                                   args.dataset,
                                   args.train_augment,
                                   args.valid_augment)

        # output parameter settings
        os.makedirs(args.model_save_dir, exist_ok=True)
        save_args(args, args.model_save_dir)
        predictor.save_args(os.path.join(args.model_save_dir, 'model.json'))
        normalizer.summary(os.path.join(args.model_save_dir, 'normalize.json'))
        augmentor.summary(os.path.join(args.model_save_dir, 'augment.json'))

        # start train
        train_phase(predictor, train, valid, args)


if __name__ == '__main__':
    main()
