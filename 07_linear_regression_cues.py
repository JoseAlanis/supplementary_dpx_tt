"""
==============================
Fit linear model to cue epochs
==============================

Mass-univariate analysis of cue evoked activity.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np
import pandas as pd

from scipy import stats

from mne import grand_average
from mne.viz import plot_compare_evokeds

from sklearn.metrics import r2_score

from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
from mne.channels import find_ch_connectivity
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne.viz import plot_topomap, plot_compare_evokeds, tight_layout
from mne import read_epochs, combine_evoked, find_layout

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat

# dicts for storing individual sets of epochs/ERPs
cues = dict()

# baseline to be applied
baseline = (-0.300, -0.050)

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

    # extract a and b epochs (only those with correct responses)
    # and apply baseline
    cues['subj_%s' % subj] = cue_epo['Correct A', 'Correct B']
    cues['subj_%s' % subj].apply_baseline(baseline).crop(tmin=-0.500)

###############################################################################
# 2) linear model parameters
# use first subject as generic information template for results
generic = cues['subj_%s' % subjects[0]].copy()

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.). This is needed for creating an homologous mne.Epochs object
# containing the results of the linear regression in an eeg-like format
# (i.e., channels x times points).
epochs_info = generic.info
n_channels = len(epochs_info['ch_names'])
n_times = len(generic.times)

# also save times first time-point in data
times = generic.times
tmin = generic.tmin

# subjects
subjects = list(cues.keys())

# independent variables to be used in the analysis (i.e., predictors)
predictors = ['intercept', 'cue']

# number of predictors
n_predictors = len(predictors)

###############################################################################
# 3) initialise place holders for the storage of results
# place holders for bootstrap samples
betas = np.zeros((len(predictors),
                  len(cues.values()),
                  n_channels * n_times))

# dicts for results evoked-objects
betas_evoked = dict()
t_evokeds = dict()
r2_evoked = dict()

###############################################################################
# 4) Fit linear model on a subject level

for subj_ind, subject in enumerate(cues.values()):

    # 4.1) create subject design matrix using epochs metadata
    metadata = subject.metadata.copy()

    # only keep predictor columns
    design = metadata[predictors]
    # add intercept (constant) to design matrix
    design = design.assign(intercept=1)

    design = pd.get_dummies(design, drop_first=True)


