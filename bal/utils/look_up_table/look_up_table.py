import os
from copy import copy

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap


class LookUpTable:
    def __init__(self,
                 name=None,
                 new_colors=None,
                 filename=None):

        assert name is not None or new_colors is not None or filename is not None

        self.name = name
        self.new_colors = new_colors
        self.filename = filename

        # create a new look_up_table using the given array
        if self.new_colors is not None:
            self.lut = ListedColormap(self.new_colors)

        # get look_up_table from indicated files
        elif self.filename is not None:
            self.set_lut_from_file(self.filename)

        # use pre-defined look_up_table
        else:
            self.set_lut_from_name(self.name)

    def __call__(self, idx):
        return self.lut(idx)

    def __len__(self):
        return self.lut.N

    @staticmethod
    def load_csv(filename):
        df = pd.read_csv(filename)
        lut = []
        for i in range(len(df)):
            if 'opacity' in df.keys():
                arr = [df['color_r'][i], df['color_g'][i], df['color_b'][i], df['opacity'][i]]
            else:
                arr = [df['color_r'][i], df['color_g'][i], df['color_b'][i]]
            lut.append(arr)
        return lut

    def copy(self):
        return copy(self.lut)

    def set_lut_from_name(self, name):
        if name in _DEFINED_COLOR.keys():
            arr = _DEFINED_COLOR[name]
            self.lut = ListedColormap(arr)
        else:
            self.lut = cm.get_cmap(name)

    def set_lut_from_file(self, filename):
        _, ext = os.path.splitext(filename)
        if ext == '.csv':
            lut = self.load_csv(filename)
        else:
            raise NotImplementedError('Currently, only csv files are supported.')

        self.lut = ListedColormap(np.array(lut))

    def change_opacity(self):
        raise NotImplementedError()

    def save_to_csv(self, filename):
        raise NotImplementedError()


_DEFINED_COLOR = {
    'muscles': np.array((
        [0, 0, 0, 1],  # background
        [1, 1, 1, 1],  # pelvis
        [1, 1, 1, 1],  # femur
        [0, 1, 1, 1],
        [0.75, 1, 0.25, 1],
        [1, 1, 0, 1],
        [0, 1, 0, 1],
        [1, 0.5, 0.5, 1],
        [1, 0.5, 0.5, 1],
        [0.5, 0, 0.5, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 0.5, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 0.5, 0],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [0.5, 0, 0.5, 1],
        [0, 1, 0, 1],
        [0.5, 0.5, 0.5, 1],
        [0, 1, 0.5, 1],
        [0, 0.5, 0, 1],
    ), dtype=object),
    'hip_implant': np.array((
        [0, 0, 0, 0],  # background
        [1, 0, 0, 1],  # cup
        [0, 1, 0, 1],  # head
        [0, 0, 1, 1],  # stem
    ), dtype=object),
    'quadriceps': np.array((
        [0, 0, 0, 0],  # background
        [1, 0, 0, 1],  # rectus femoris
        [0, 0, 1, 1],  # vastus intermedius
        [0.5, 0, 0.5, 1],  # vastus lateralis
        [0, 1, 0, 1],  # vastus medialis
    ), dtype=object),
    'bones': np.array((
        [0, 0, 0, 1],  # background
        [0, 1, 0, 1],  # pelvis
        [1, 1, 0, 1],  # femur
    ), dtype=object)
}

if __name__ == '__main__':

    config = {'name': 'hip_implant'}
    lut = LookUpTable(**config)

    print(lut.name)
    for i in range(100):
        print(lut(i))
