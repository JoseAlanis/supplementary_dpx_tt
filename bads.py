# 2) Check if there are flat channels
import warnings

import numpy as np

from scipy.stats import median_absolute_deviation as mad
from sklearn.preprocessing import normalize

from mne.io.base import BaseRaw

def find_flat_channels(inst, picks='eeg', mad_threshold=1, std_threshold=1,
                       channels=None):

    kwargs = {pick: True for pick in [picks]}

    if isinstance(inst, BaseRaw):
        # only keep data from desired channels
        inst = inst.copy().pick_types(**kwargs)
        dat = inst.get_data() * 1e6  # to microvolt
        channels = inst.ch_names
    elif isinstance(inst, np.ndarray):
        if not channels:
            raise ValueError('If "inst" is not an instance of BaseRaw a '
                             'list '
                             'of channel names must be provided')
        dat = inst

    else:
        raise ValueError('inst must be an instance of BaseRaw or a numpy '
                         'array')

    # compute estimates of channel activity
    mad_flats = mad(dat, scale=1, axis=1) < mad_threshold
    std_flats = np.std(dat, axis=1) < std_threshold

    # flat channels identified
    flats = np.argwhere(np.logical_or(mad_flats, std_flats))
    flats = np.asarray([channels[int(flat)] for flat in flats])

    if len(flats) > (len(channels) / 2):
        warnings.warn('Too many channels have been identified as "flat"! '
                      'Make sure the input values in "inst" are provided on a '
                      'volt scale. Otherwise try choosing a meaningful '
                      'threshold for identification.')

    return flats