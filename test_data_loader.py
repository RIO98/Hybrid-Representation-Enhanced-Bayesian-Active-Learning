from collections import OrderedDict
from functools import partial

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bal.dataloader import SliceImageDataset
from bal.data import DataAugmentor
from bal.data import Normalizer
from bal.data import Distort2D
from bal.data import Clip2D

if __name__ == '__main__':
    # setup augmentor
    augmentor = DataAugmentor(n_dim=2)
    # augmentor.add(Flip2D(probability=0.9, axis=2))
    # augmentor.add(Affine2D(probability=0.9,
    #                        rotation=10.,
    #                        translate=(25., 25.),
    #                        shear=np.pi / 8.,
    #                        fill_mode=('constant', 'nearest'),
    #                        zoom=(0.75, 1.25),
    #                        cval=(0., 0.),
    #                        interp_order=(1, 0)))
    augmentor.add(Distort2D(probability=1.0,
                            alpha=(75, 125),
                            sigma=10,
                            order=(3, 0)))
    # augmentor.add(RandomErasing2D(probability=0.9,
    #                               size=(0.02, 0.2),
    #                               ratio=(1.0, 0.3),
    #                               value_range=(0, 1)))

    # setup normalizer
    normalizer = Normalizer(n_dim=2)
    normalizer.add(Clip2D((0, 1000)))

    root = r'D:\Database\dataset\MR_Quad\256'
    train_patients = np.loadtxt(r'D:\Project\MRI2CT\Bayesian_unet\scripts\exp\22\seed_0\4layer\random\id-list_trial-1_training-0.txt', dtype=str)
    train_patients = [r.replace('/', r'\\') for r in train_patients]

    image_ext = 'image_*.mha'
    label_ext = 'muscle_label_*.mha'
    dtypes = OrderedDict({'image': np.float32, 'label': np.int64})
    exts = OrderedDict({'image': image_ext, 'label': label_ext})
    class_list = list(range(5))
    getter = partial(SliceImageDataset, root=root, classes=class_list,
                     dtypes=dtypes, exts=exts, normalizer=normalizer)
    train_filenames = OrderedDict({
        'image': '{root}/{patients}',
        'label': '{root}/{patients}',
    })

    train = getter(patients=train_patients, filenames=train_filenames, augmentor=augmentor)
    train_loader = iter(DataLoader(train, 4, shuffle=False, num_workers=4))

    for i in range(10):
        a, b = next(train_loader)

    for i in range(5):
        a, b = next(train_loader)
        print(a.size(), b.size())
        # Convert the tensor to a NumPy array and transpose the dimensions to (256, 256, 3)
        array = a.numpy().transpose(0, 2, 3, 1)

        # Display the image using matplotlib
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(array[1], cmap='gray')
        axs[1].imshow(b.numpy()[1], cmap='gray')
        plt.show()
