"""
=======================
Time frequency analysis
=======================

Fit morlet-wavelet decomposition on sigle subject eeg data.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np
import matplotlib.pyplot as plt

from mne import read_epochs, grand_average, combine_evoked
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat, n_jobs

# wavelet parameters and channels of interest
tmax = 2.45
decim = 3
baseline = (-1.0, -0.5)
freqs = np.logspace(*np.log10([1, 40]), num=20)
n_fq = len(freqs)
n_cycles = np.logspace(*np.log10([1, 10]), num=n_fq)
channels = ['PO7', 'PO8',
            'Pz', 'CPz', 'CP1',
            'FCz', 'FC1',
            'C1', 'C3',
            'AF4', 'AF3']

# freqs and labels for plots
ticks = np.unique(np.linspace(1, n_fq, 10).round()).astype('int')
tick_vals = freqs[np.unique(np.linspace(0, n_fq - 1, 10).round()).astype(
    'int')].round(2)

# place holders for tfr results
tfr_cue_a = dict()
tfr_cue_b = dict()

# place holders for power values
power_a = dict()
power_b = dict()

###############################################################################
# 1) loop through subjects and compute ERPs for A and B cues
for subj in subjects:

    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Loading epochs for subject %s' % subj +
          LoggingFormat.END)

    # import the output from previous processing step
    input_file = fname.output(subject=subj,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    cue_epo = read_epochs(input_file, preload=True)

    # # extract A and B epochs (only those with correct responses)

    # fit wavelet transform to cue A epochs
    tfr_epochs_A = tfr_morlet(cue_epo['Correct A'].crop(tmax=tmax),
                              freqs,
                              n_cycles=n_cycles,
                              decim=decim,
                              return_itc=False,
                              average=False,
                              n_jobs=n_jobs)

    # apply baseline and shorten the segment
    tfr_epochs_A = tfr_epochs_A.apply_baseline(mode='ratio', baseline=baseline)

    # fit wavelet transform to cue B epochs
    tfr_epochs_B = tfr_morlet(cue_epo['Correct B'].crop(tmax=tmax),
                              freqs,
                              n_cycles=n_cycles,
                              decim=decim,
                              return_itc=False,
                              average=False,
                              n_jobs=n_jobs)

    # apply baseline and shorten the segment
    tfr_epochs_B = tfr_epochs_B.apply_baseline(mode='ratio', baseline=baseline)

    # store results in dict (only keep shorter epochs )
    tfr_cue_a['subj_%s' % subj] = tfr_epochs_A.crop(tmin=-0.5)
    tfr_cue_b['subj_%s' % subj] = tfr_epochs_B.crop(tmin=-0.5)

    del tfr_epochs_A, tfr_epochs_B

###############################################################################
# 2) Mean of epochs TFR

# compute mean time-frequency representation for cue A
cue_a_power = [tfr_cue_a['subj_%s' % subj].average() for subj in subjects]
# compute mean time-frequency representation for cue B
cue_b_power = [tfr_cue_b['subj_%s' % subj].average() for subj in subjects]

###############################################################################
# 3) transform values to decibel

# A-cue power
for i in cue_a_power:
    i.data = 10 * np.log10((i.data * i.data.conj()).real)
# B-cue power
for i in cue_b_power:
    i.data = 10 * np.log10((i.data * i.data.conj()).real)

###############################################################################
# 4) compute grand averages

# mean of signal
average_tfr_a = grand_average(cue_a_power)
average_tfr_b = grand_average(cue_b_power)

# compute difference B-A
difference_ba = combine_evoked([average_tfr_b, average_tfr_a], weights=[1, -1])

###############################################################################
# 5) create plot

# variables for custom plot appearance
topomap_args = dict(outlines='head', sensors=False, vmin=-6.5, vmax=6.5)
title = 'TFR difference B - A (mean of 64 EEG channels)'
# create plot
fig = difference_ba.plot_joint(timefreqs=((0.2, 6.0),
                                          (0.3, 6.0),
                                          (0.6, 12.0),
                                          (0.8, 1.5),
                                          (1.1, 14.0),
                                          (2.25, 12.0)),
                               vmin=-6.0, vmax=6.0,
                               title=title,
                               topomap_args=topomap_args,
                               show=False)
fig.axes[0].axvline(x=0.0, ymin=0.0, ymax=40.0,
                    color='black', linestyle='dashed', linewidth=0.8)
fig.axes[0].set_yticks(tick_vals, minor=False)
fig.axes[0].set_xticks(list(np.arange(-0.5, 2.5, 0.25)), minor=False)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(1.0, 40.0)
fig.axes[0].spines['bottom'].set_bounds(-0.5, 2.5)
fig.axes[-1].set_title('dB')
fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1.15, h * 1.15)
fig_name = fname.figures + '/Ave_TFR_diff_BA.pdf'
fig.savefig(fig_name, dpi=300)

###############################################################################
# 6) Run cluster permutation test for single electrodes of interest

# clean some memory space
del cue_a_power, cue_b_power, average_tfr_a, average_tfr_b

# loop through subjects and extract relevant electrodes
for subj in subjects:

    for channel in channels:

        ix = tfr_cue_a['subj_%s' % subj].ch_names.index(channel)

        temp_power_a = tfr_cue_a['subj_%s' % subj].data[:, ix, :, :]
        temp_power_b = tfr_cue_b['subj_%s' % subj].data[:, ix, :, :]

        if subj == subjects[0]:
            power_a['%s' % channel] = temp_power_a
            power_b['%s' % channel] = temp_power_b
        else:
            power_a['%s' % channel] = np.concatenate(
                (power_a['%s' % channel], temp_power_a), axis=0)

            power_b['%s' % channel] = np.concatenate(
                (power_b['%s' % channel], temp_power_b), axis=0)

        del temp_power_a, temp_power_b

# loop through channels of interest and test the difference between B and A cues
for channel in channels:

    # run cluster permutations test
    threshold = 10.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([power_b[channel], power_a[channel]],
                                 n_permutations=10000, threshold=threshold,
                                 tail=0, n_jobs=n_jobs, out_type='mask')

    # Create new stats image with only significant clusters
    T_obs_plot = np.zeros_like(T_obs, dtype=bool)

    for a_lev in [0.05, 0.01]:

        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= a_lev:
                T_obs_plot[c] = True

        # variables for plot
        # channel in question
        ix = tfr_cue_a['subj_%s' % subjects[0]].ch_names.index(channel)
        # min max values
        dB_max = np.abs(difference_ba.data[ix, ...].max())
        dB_min = np.abs(difference_ba.data[ix, ...].min())
        v_scale = np.max((dB_max, dB_min))
        v_scale = np.ceil(v_scale)
        # x and y values
        times = tfr_cue_a['subj_%s' % subjects[0]].times
        x = times.copy()
        y = freqs
        y[0] = 1
        X, Y = np.meshgrid(x, y)

        # create plot
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
        difference_ba.plot(picks=[channel],
                           mask=T_obs_plot,
                           mask_style='mask',
                           mask_cmap='RdBu_r',
                           mask_alpha=0.70,
                           axes=ax,
                           vmin=np.negative(v_scale),
                           vmax=v_scale,
                           show=False,
                           title='Difference B-A at electrode %s' % channel)

        # add box around significant clusters
        ax.contour(X, Y, T_obs_plot, colors=["k"], linewidths=[1],
                   corner_mask=False, antialiased=False, levels=[.5],
                   origin='lower')
        w, h = fig.get_size_inches()
        ax.axvline(x=0, color='black', linestyle='--')
        fig.axes[-1].set_title('dB')
        fig.axes[0].set_yticks(tick_vals, minor=False)
        fig.axes[0].set_xticks(list(np.arange(-0.5, 2.5, 0.25)), minor=False)
        fig.axes[0].spines['top'].set_visible(False)
        fig.axes[0].spines['right'].set_visible(False)
        fig.axes[0].spines['left'].set_bounds(1.0, 40.0)
        fig.axes[0].spines['bottom'].set_bounds(-0.5, 2.5)

        # adjust borders
        fig.subplots_adjust(
            left=0.10, right=1.0, bottom=0.15, top=0.90,
            wspace=0.3, hspace=0.25)
        fig_name = fname.figures + '/%s_TFR_clusters_BA_p%s.pdf' % (channel,
                                                                    a_lev)
        # save plot
        fig.savefig(fig_name, dpi=300)
