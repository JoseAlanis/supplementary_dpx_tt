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

import numpy as np

from mne import create_info
from mne.io import read_raw_fif, RawArray

from scipy.stats import median_absolute_deviation as MAD

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

ref_signal = np.nanmedian(eeg_signal, axis=0)

# mask
mask = [0] * eeg_signal.shape[0]


[MAD(eeg_signal[i, :]) * 1e6 for i in range(eeg_signal.shape[0])]

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


