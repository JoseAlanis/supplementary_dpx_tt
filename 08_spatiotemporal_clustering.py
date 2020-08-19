"""
=====================================
Compute T-map for effect of condition
=====================================

Mass-univariate analysis of cue evoked activity.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

from mne.stats.cluster_level import _setup_connectivity, _find_clusters
from mne.channels import find_ch_connectivity
from mne import read_epochs

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat

# load individual beta coefficients
betas = np.load(fname.results + '/subj_betas_cue_m250.npy')

###############################################################################
# 1) import epochs to use as template

# import the output from previous processing step
input_file = fname.output(subject=subjects[0],
                          processing_step='cue_epochs',
                          file_type='epo.fif')
cue_epo = read_epochs(input_file, preload=True)
cue_epo = cue_epo['Correct A', 'Correct B'].copy()
cue_epo = cue_epo.crop(tmin=-0.25, tmax=2.45)

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.).
epochs_info = cue_epo.info
n_channels = len(epochs_info['ch_names'])
n_times = len(cue_epo.times)
times = cue_epo.times

###############################################################################
# 2) compute bootstrap confidence interval for phase-coherence betas and
# t-values

# set random state for replication
random_state = 42
random = np.random.RandomState(random_state)

# number of random samples
boot = 10000

# place holders for bootstrap samples
cluster_H0 = np.zeros(boot)
f_H0 = np.zeros(boot)

# setup connectivity
n_tests = betas.shape[1]
connectivity, ch_names = find_ch_connectivity(epochs_info, ch_type='eeg')
connectivity = _setup_connectivity(connectivity, n_tests, n_times)

# threshold parameters for clustering
threshold = dict(start=0.2, step=0.2)

# run bootstrap for regression coefficients
for i in range(boot):

    # log progress
    print(LoggingFormat.PURPLE +
          LoggingFormat.BOLD +
          'Running bootstrap sample %s of %s' % (i, boot) +
          LoggingFormat.END)

    # extract random subjects from overall sample
    resampled_subjects = random.choice(range(betas.shape[0]),
                                       betas.shape[0],
                                       replace=True)
    # resampled betas
    resampled_betas = betas[resampled_subjects, :]

    # compute standard error of bootstrap sample
    se = resampled_betas.std(axis=0) / np.sqrt(resampled_betas.shape[0])

    # center re-sampled betas around zero
    resampled_betas -= betas.mean(axis=0)

    # compute t-values for bootstrap sample
    t_vals = resampled_betas.mean(axis=0) / se
    # transfrom to f-values
    f_vals = t_vals ** 2
    # save max f-value
    f_H0[i] = f_vals.max()

    # transpose for clustering
    t_vals = t_vals.reshape((n_channels, n_times))
    t_vals = np.transpose(t_vals, (1, 0))
    t_vals = t_vals.ravel()

    # compute clustering on squared t-values (i.e., f-values)
    clusters, cluster_stats = _find_clusters(t_vals,
                                             t_power=1,
                                             threshold=threshold,
                                             connectivity=connectivity,
                                             tail=0)

    # save max cluster mass. Combined, the max cluster mass values
    # computed on the basis of the bootstrap samples provide an approximation
    # of the cluster mass distribution under H0
    if len(clusters):
        cluster_H0[i] = cluster_stats.max()
    else:
        cluster_H0[i] = np.nan

##############################################################################
# 3) Save results of bootstrap procedure

# save f-max distribution
np.save(fname.results + '/f_H0_10000b_2t_m250.npy', f_H0)
# save cluster mass distribution
np.save(fname.results + '/cluster_H0_10000b_2t_m250.npy', cluster_H0)
