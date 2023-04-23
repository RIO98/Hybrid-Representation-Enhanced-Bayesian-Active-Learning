# Code adapted from https://github.com/yuta-hi/remove-island/blob/master/refine_label/remove_island.py

import math
from typing import List, Tuple

import cc3d
import numpy as np
import tqdm
from scipy import ndimage as ndi
from skimage.measure._regionprops import _RegionProperties

_supported_metrics = [
    'area',
    'bbox_area',
    'convex_area',
    'filled_area']


def _labeling(object_label: np.ndarray, connectivity: int) -> np.ndarray:
    """
    Grouping pixels that are connected based on the specified connectivity (Connected Components Labeling (CCL)).

    :param object_label: np.ndarray, input binary/multilabel Connected Components Labeling image
    :param connectivity: int, the connectivity defining the neighborhood
    :return: np.ndarray, labeled image
    """
    return cc3d.connected_components(object_label, connectivity=connectivity)


def _regionprop_nd(object_label: np.ndarray, island_label: np.ndarray, metric: str, cache: bool = True) -> Tuple[
    List[List[float]], List[List[int]]]:
    """
    Calculate region properties for the given objects and islands.

    :param object_label: np.ndarray, input binary image with labeled objects
    :param island_label: np.ndarray, labeled image with connected components
    :param metric: str, metric to be calculated for region properties
    :param cache: bool, optional, use caching for better performance (default: True)
    :return: tuple of lists, (prop_table, index_table)
    """

    # make the table of prop and its index
    n_object = np.max(object_label) + 1
    n_island = np.max(island_label) + 1

    prop_table = [[] for _ in range(n_object)]
    index_table = [[] for _ in range(n_object)]

    islands = ndi.find_objects(island_label)

    for i, sl in tqdm.tqdm(enumerate(islands), total=len(islands),
                           ncols=80, leave=False, desc='regionprop'):

        if sl is None:
            continue

        props = _RegionProperties(sl,
                                  i + 1,
                                  island_label,
                                  None,
                                  cache_active=cache)

        prop = getattr(props, metric)

        start = props.coords[0]
        start = [slice(s, s + 1) for s in start]
        obj = object_label[tuple(start)]
        obj = int(np.squeeze(obj))

        prop_table[obj].append(prop)
        index_table[obj].append(i + 1)

    return prop_table, index_table


def remove_island(object_label: np.ndarray, noise_ratio: float = 5., connectivity: int = 6,
                  metric: str = 'area', cval: int = 0, only_largest: bool = False) -> np.ndarray:
    """
    Remove small connected components (islands) from the input image.

    :param object_label: np.ndarray, input binary image with labeled objects
    :param noise_ratio: float, optional, percentile threshold for noise removal (default: 5.)
    :param connectivity: int, optional, connectivity defining the neighborhood (default: 6)
    :param metric: str, optional, metric for region properties (default: 'area')
    :param cval: int, optional, value to replace removed objects with (default: 0)
    :param only_largest: bool, optional, keep only the largest object (default: False)
    :return: np.ndarray, processed image with small islands removed
    """

    ret = object_label.copy()
    ret_shape = object_label.shape

    # find the connected components, and measure the area
    if metric not in _supported_metrics:
        raise KeyError('metric should be in (' + ','.join(_supported_metrics) + ')')

    island_label = _labeling(object_label, connectivity)
    area_table, index_table = _regionprop_nd(object_label, island_label, metric)

    # remove small islands
    background = []

    for areas, indices in zip(area_table, index_table):

        if len(areas) == 0:
            continue

        areas_sort_ind = np.argsort(areas)[::-1]
        areas = np.asarray(areas)[areas_sort_ind]
        indices = np.asarray(indices)[areas_sort_ind]

        if only_largest:
            threshold = 0

        else:
            histogram = []
            for i, area in enumerate(areas):
                histogram.extend([i] * area)

            threshold = math.ceil(np.percentile(histogram, 100.0 - noise_ratio))

        background = np.concatenate([background, indices[threshold + 1:]])

    mask = np.in1d(island_label.ravel(), background)
    mask = mask.reshape(ret_shape)
    ret[mask] = cval

    return ret
