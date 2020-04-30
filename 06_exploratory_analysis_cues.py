"""
==================================
Exploratory analysis of cue epochs
==================================

Compute descriptive statistics and exploratory analysis plots
for cue locked epochs.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import pandas as pd
import numpy as np

from mne import read_epochs, combine_evoked, grand_average
from mne.viz import plot_compare_evokeds

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat

epochs_a_cue = dict()
epochs_b_cue = dict()

erps_a_cue = dict()
erps_b_cue = dict()

for subject in subjects:

    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Loading epochs for subject %s' % subject +
          LoggingFormat.END)

    # import the output from previous processing step
    input_file = fname.output(subject=subject,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    cue_epochs = read_epochs(input_file, preload=True)

    # extract a and b epochs (only those with correct responses)
    a_epochs = cue_epochs['Correct A']
    b_epochs = cue_epochs['Correct B']

    # apply baseline
    epochs_a_cue['subj_%s' % subject] = a_epochs.apply_baseline((-0.3, -0.05))
    epochs_b_cue['subj_%s' % subject] = b_epochs.apply_baseline((-0.3, -0.05))

    # apply baseline
    erps_a_cue['subj_%s' % subject] = epochs_a_cue['subj_%s' % subject].average()
    erps_b_cue['subj_%s' % subject] = epochs_b_cue['subj_%s' % subject].average()


# weights = np.repeat(1 / len(erps_a_cue), len(erps_a_cue))
# ga_a_cue = combine_evoked(list(erps_a_cue.values()),
#                           weights=list(weights))

ga_a_cue = grand_average(list(erps_a_cue.values()))
ga_b_cue = grand_average(list(erps_b_cue.values()))

# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-10, 10]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [.100, .180, .310, .500, .680, 1.00, 2.45]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='s',
                    vmin=8, vmax=-8,
                    average=0.05,
                    outlines='skirt')

ga_a_cue.plot_joint(ttp, ts_args=ts_args, topomap_args=topomap_args)
ga_b_cue.plot_joint(ttp, ts_args=ts_args, topomap_args=topomap_args)

plot_compare_evokeds({'A': ga_a_cue, 'B': ga_b_cue},
                     picks=['Pz'],
                     invert_y=True,
                     ylim=dict(eeg=[-10, 5]),
                     colors={'A': 'k', 'B': 'crimson'})

