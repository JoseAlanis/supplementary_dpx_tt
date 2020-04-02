"""
=============================================================
Extract segments of the data recorded during task performance
=============================================================

Segments that were recorded during the self-paced breaks (in between
experimental blocks) will be dropped.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap('inferno')

import numpy as np

from mne import create_info
from mne.io import read_raw_fif, RawArray

from scipy.stats import median_absolute_deviation as mad
from sklearn.preprocessing import normalize

# All parameters are defined in config.py
from config import fname, n_jobs, parser

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Converting subject %s to BIDS' % subject)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='task_blocks',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

###############################################################################
# 2)

raw_copy = raw.copy()

eeg_signal = raw_copy.get_data(picks='eeg')

# find bad channels by deviation (high variability in amplitude)
ref_signal = np.nanmedian(eeg_signal, axis=0)

temp_eeg = eeg_signal - ref_signal

# mean absolute deviation
mad_scores = [mad(temp_eeg[i, :], scale=1) for i in range(temp_eeg.shape[0])]
# compute robust z-scores
robust_z_scores_dev = 0.6745 * (mad_scores - np.nanmedian(mad_scores)) / \
                          mad(mad_scores, scale=1)

# channels identified by deviation criterion
bads_by_dev = [raw_copy.ch_names[i] for i in np.where(robust_z_scores_dev >
                                                  3.5)[0]]

# plot results
z_colors = normalize(np.abs(robust_z_scores_dev).reshape((1,
                                                          robust_z_scores_dev.shape[0]))).ravel()

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
fig, ax = plt.subplots(figsize=(5, 15))
for i in range(robust_z_scores_dev.shape[0]):
    ax.axvline(x=5.0, ymin=-5.0, ymax=5.0,
               color='crimson', linestyle='dotted', linewidth=.8)
    ax.text(5.0, -2.0, 'crit. Z-score',  fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            color='crimson', bbox=props)
    ax.barh(i, np.abs(robust_z_scores_dev[i]), 0.9, color=cmap(z_colors[i]))
    ax.text(np.abs(robust_z_scores_dev[i]) + 0.25, i, raw.info['ch_names'][i],
            ha='center', va='center', fontsize=9)
ax.set_xlim(0, int(robust_z_scores_dev.max()+2))
plt.title('EEG channel deviation')
plt.xlabel('Abs. Z-Score')
plt.ylabel('Channels')
plt.show()





def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)


mad_scores_2 = [np.abs(eeg_signal[1, i] - np.nanmedian(eeg_signal[1, :])) for
                       i in range(eeg_signal.shape[1])]

chan_dev = [0.7413 * iqr(eeg_signal[i, :]) for i in range(eeg_signal.shape[0])]
channel_deviationSD = 0.7413 * iqr(chan_dev)
channel_deviationMedian = np.nanmedian(chan_dev)

robust_channel_deviation = np.divide(np.subtract(chan_dev, channel_deviationMedian), channel_deviationSD)

robust_z_scores_dev = (mad_scores - np.nanmedian(mad_scores)) / mad(mad_scores)

# mask
mask = [0] * eeg_signal.shape[0]

# TEST
ref_signal = ref_signal.reshape(1, ref_signal.shape[0])

info = create_info(
    ch_names=['REF'],
    ch_types=['eeg'],
    sfreq=raw_copy.info['sfreq'])

custom_raw = RawArray(ref_signal, info)
custom_raw.info['highpass'] = raw_copy.info['highpass']
custom_raw.info['lowpass'] = raw_copy.info['lowpass']
custom_raw.info['line_freq'] = raw_copy.info['line_freq']
raw_copy.add_channels([custom_raw])

# raw_temp = raw_copy()
# sig_temp = raw_copy() - ref_signal
#
# while True
#   detect bads in sig_temp
#   if no bads
#       break
#   elif bads
#       raw_temp = raw_copy()
#       raw_temp[bads] = bads
#       raw_temp intepolate bads
#   else


# mean = mean of raw after interpolateion of bads
# raw_temp = raw_copy - mean


