# 2) Check if there are flat channels
import warnings

import numpy as np

from scipy.stats import median_absolute_deviation as mad
from sklearn.preprocessing import normalize

from mne.io.base import BaseRaw


def find_bad_channels(inst, picks='eeg',
                      method='uncorrelation',
                      mad_threshold=1,
                      std_threshold=1,
                      r_threshold=0.4,
                      percent_threshold=0.1,
                      time_step=1.0,
                      max_iter = 4,
                      return_ref=False,
                      channels=None):

    kwargs = {pick: True for pick in [picks]}

    if isinstance(inst, BaseRaw):
        # only keep data from desired channels
        inst = inst.copy().pick_types(**kwargs)
        dat = inst.get_data() * 1e6  # to microvolt
        channels = inst.ch_names
    elif isinstance(inst, np.ndarray):
        if not channels:
            raise ValueError('If "inst" is not an instance of BaseRaw a list '
                             'of channel names must be provided')
        dat = inst

    else:
        raise ValueError('inst must be an instance of BaseRaw or a numpy array')

    # make sure method argumants are in a list
    if not isinstance(method, list):
        method = [method]

    # place holder for results
    bad_channels = dict()

    # 1) find channels with zero or near zero activity
    if 'flat' in method:
        # compute estimates of channel activity
        mad_flats = mad(dat, scale=1, axis=1) < mad_threshold
        std_flats = np.std(dat, axis=1) < std_threshold

        # flat channels identified
        flats = np.argwhere(np.logical_or(mad_flats, std_flats))
        flats = np.asarray([channels[int(flat)] for flat in flats])

        # warn user if too many channels were identified as flat
        if len(flats) > (len(channels) / 2):
            warnings.warn('Too many channels have been identified as "flat"! '
                          'Make sure the input values in "inst" are provided '
                          'on a volt scale. '
                          'Otherwise try choosing another (meaningful) '
                          'threshold for identification.')

        bad_channels.update(flat=flats)

    # 3) find channels with high deviation scores
    if 'deviation' in method
        inst_copy = inst.copy()
        eeg = inst_copy.get_data()

        # get robust estimate of central tendency (i.e., the median)
        ref = np.nanmedian(eeg, axis=0)
        # remove reference from eeg signal
        eeg -= ref

        noisy = []
        iterations = 0
        while True:
            # find bad channels by deviation (high variability in amplitude)
            # mean absolute deviation (MAD) scores for each channel
            mad_scores = \
                [mad(eeg[i, :], scale=1) for i in range(eeg.shape[0])]

            # compute robust z-scores for each channel
            rz_scores_dev = \
                0.6745 * (mad_scores - np.nanmedian(mad_scores)) / mad(
                    mad_scores,
                    scale=1)

            # channels identified by deviation criterion
            bad_deviation = \
                [channels[i] for i in np.where(np.abs(rz_scores_dev) > 5.0)[0]]

            # save channels exceeding deviation threshold
            noisy.extend(bad_deviation)

            if ((iterations > 1) and (not bad_deviation or set(bad_deviation) == set(noisy))
                    or iterations > max_iter):
                break

            if bad_deviation:
                # interpolate any channels that showed deviations scores
                # above threshold
                inst_copy.info['bads'] = list(set(noisy))
                inst_copy.interpolate_bads(mode='accurate')

            # recompute referenced eeg signal now using the mean of all channels
            # as reference
            eeg = inst_copy.get_data()
            ref = np.nanmean(eeg, axis=0)
            eeg -= ref

            iterations = iterations + 1

        bad_channels.update(deviation=np.asarray(noisy))

        # if desired also return the estimate of the robust average reference
        if return_ref:
            bad_channels.update(robust_reference=ref)

    # 3) find channels with low correlation to other channels
    if 'uncorrelation' in method:
        # based on the length of the provided data,
        # determine how many size and amount of time windows
        # for analyses
        corr_frames = time_step * inst.info['sfreq']
        corr_window = np.arange(0, corr_frames)
        n = corr_window.shape[0]
        corr_offsets = \
            np.arange(0, (len(inst.times) - corr_frames), corr_frames)
        w_correlation = corr_offsets.shape[0]

        # placeholder for correlation coefficients
        channel_correlations = np.ones((w_correlation, len(inst.ch_names)))

        # cut the data into windows
        x_bp_window = dat[: len(channels), : n * w_correlation]
        x_bp_window = x_bp_window.reshape(len(channels), n, w_correlation)

        # compute (pearson) correlation coefficient across channels
        # (for each channel and analysis time window, take the absolute of
        # the 98th percentile of the correlations with the other channels as
        # a measure of how well correlated that channel and the other channels
        # are correlated)

        for k in range(w_correlation):
            eeg_portion = x_bp_window[:, :, k]
            window_correlation = np.corrcoef(eeg_portion)
            abs_corr = \
                np.abs((window_correlation - np.diag(np.diag(window_correlation))))  # noqa: E501
            channel_correlations[k, :] = np.percentile(abs_corr, 98, axis=0)

        # check which channels correlate badly with the other channels (i.e.,
        # are below correlation threshold) in a certain fraction of windows
        # (bad_time_threshold)
        thresholded_correlations = channel_correlations < r_threshold
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_idxs_bool = frac_bad_corr_windows > percent_threshold
        bad_idxs = np.argwhere(bad_idxs_bool)
        uncorrelated_channels = [channels[int(bad)] for bad in bad_idxs]

        bad_channels.update(uncorrelation=np.asarray(uncorrelated_channels))

    return bad_channels
