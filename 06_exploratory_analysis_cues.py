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

from mne import read_epochs

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat

epochs_a_cue = dict()
epochs_b_cue = dict()

for subject in subjects:

    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Finding and removing bad components for subject %s' % subject +
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
    epochs_b_cue['subj_%s' % subject] = a_epochs.apply_baseline((-0.3, -0.05))

