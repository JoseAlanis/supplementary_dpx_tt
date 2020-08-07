
import os.path as op

import numpy as np
from numpy.random import randn

from scipy import stats as stats

from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_alignment, plot_cov
from mne import read_epochs, make_forward_solution, sensitivity_map, \
    compute_covariance, read_source_spaces, compute_source_morph, \
    spatial_src_adjacency

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat, n_jobs

fs_dir = fetch_fsaverage(verbose=True)

subjects_dir = op.dirname(fs_dir)
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


epochs = dict()

###############################################################################
# 1) loop through subjects and compute ERPs for A and B cues
for subj in subjects:
    # import the output from previous processing step
    input_file = fname.output(subject=11,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    cue_epo = read_epochs(input_file, preload=True)

    epochs['subj_%s' % subj] = cue_epo['Correct A', 'Correct B']

epochs_info = epochs['subj_%s' % subjects[0]].info

plot_alignment(epochs_info,
               src=src, eeg=['original', 'projected'],
               trans=trans,
               show_axes=True, mri_fiducials=True, dig='fiducials')

fwd = make_forward_solution(epochs_info, trans=trans, src=src,
                            bem=bem, eeg=True, mindist=5.0, n_jobs=1)

eeg_map = sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[5, 50, 100]))

noise_cov = compute_covariance(epochs['subj_%s' % subjects[0]],
                               tmax=0., method=['shrunk', 'empirical'],
                               rank=None, verbose=True)
fig_cov, fig_spectra = plot_cov(noise_cov, epochs_info)

# make an MEG inverse operator
inverse_operator = make_inverse_operator(
    epochs_info, fwd, noise_cov, loose=0.2, depth=0.8)
del fwd

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"

sample_vertices = [s['vertno'] for s in inverse_operator['src']]

sfreq = 50
n_times = len(np.arange(0.0, 301, 1000/50))
n_vertices = 20484
X = np.zeros((n_vertices, n_times, len(subjects), 2))
X = np.zeros((2, len(subjects), n_vertices, n_times))


for n_subj, subj in enumerate(subjects):
    subj_epo = epochs['subj_%s' % subj]
    A_evoked = subj_epo['Correct A'].apply_baseline((-0.3, -0.05)).average()
    A_evoked.crop(tmin=-0.3, tmax=0.31)
    A_evoked.resample(sfreq, npad='auto')
    condition1 = apply_inverse(A_evoked, inverse_operator, lambda2, method)

    B_evoked = subj_epo['Correct B'].apply_baseline((-0.3, -0.05)).average()
    B_evoked.crop(tmin=-0.3, tmax=0.31)
    B_evoked.resample(sfreq, npad='auto')
    condition2 = apply_inverse(B_evoked, inverse_operator, lambda2, method)

    condition1.crop(0, None)
    condition2.crop(0, None)
    tmin = condition1.tmin
    tstep = condition1.tstep * 1000  # convert to milliseconds

    X[0, n_subj, ...] = condition1.data
    X[1, n_subj, ...] = condition2.data

X = np.abs(X)  # only magnitude
X = X[:, :, :, 1] - X[:, :, :, 0]  # make paired contrast
X = X[1, :, :, :] - X[0, :, :, :]  # make paired contrast

print('Computing adjacency.')
source = read_source_spaces(src)
adjacency = spatial_src_adjacency(source)

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [0, 2, 1])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.01
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjects) - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=8,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

print('Visualizing clusters.')

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=sample_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='both', views='lateral', subjects_dir=subjects_dir,
    time_label='temporal extent (ms)', size=(800, 800),
    smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))