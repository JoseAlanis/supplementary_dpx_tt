
import os.path as op

import numpy as np

from scipy import stats as stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne.datasets import fetch_fsaverage
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne import read_epochs, make_forward_solution, compute_covariance, \
    SourceEstimate

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat, n_jobs

fs_dir = fetch_fsaverage(verbose=True)

subjects_dir = op.dirname(fs_dir)
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

baseline = (-0.300, -0.050)

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

stcs_a = []
stcs_b = []

###############################################################################
# 1) loop through subjects and compute ERPs for A and B cues
for subj in subjects:
    # import the output from previous processing step
    input_file = fname.output(subject=subj,
                              processing_step='cue_epochs',
                              file_type='epo.fif')
    cue_epo = read_epochs(input_file, preload=True)

    a_epo = cue_epo['Correct A']
    a_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=0.20)
    b_epo = cue_epo['Correct B']
    b_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=0.20)

    a_epochs_info = a_epo.info
    b_epochs_info = b_epo.info

    equalize_epoch_counts([a_epo, b_epo])

    noise_cov_a = compute_covariance(
        a_epo, tmax=0.,
        method=['shrunk', 'empirical'], rank=None, verbose=True)
    noise_cov_b = compute_covariance(
        b_epo, tmax=0.,
        method=['shrunk', 'empirical'], rank=None, verbose=True)

    evoked_a = a_epo.average()
    evoked_b = b_epo.average()
    # evoked_a.plot(time_unit='s')
    # evoked_a.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg',
    #                       time_unit='s')
    # Show whitening:
    # evoked_a.plot_white(noise_cov_a, time_unit='s')

    fwd_a = make_forward_solution(evoked_a.info, trans=trans, src=src,
                                  bem=bem, eeg=True, mindist=5.0, n_jobs=2)
    fwd_b = make_forward_solution(evoked_b.info, trans=trans, src=src,
                                  bem=bem, eeg=True, mindist=5.0, n_jobs=2)

    # make an MEG inverse operator
    inverse_operator_a = make_inverse_operator(
       evoked_a.info, fwd_a, noise_cov_a, loose=0.2, depth=0.8)
    inverse_operator_b = make_inverse_operator(
       evoked_b.info, fwd_b, noise_cov_b, loose=0.2, depth=0.8)
    del fwd_a, fwd_b

    stc_a, residual_a = apply_inverse(evoked_a, inverse_operator_a, lambda2,
                                      method=method, pick_ori=None,
                                      return_residual=True, verbose=True)

    stc_b, residual_b = apply_inverse(evoked_b, inverse_operator_b, lambda2,
                                      method=method, pick_ori=None,
                                      return_residual=True, verbose=True)

    # store results in list
    stcs_a.append(stc_a)
    stcs_b.append(stc_b)

data = np.average([s.data for s in stcs_a], axis=0)
average_stc_a = SourceEstimate(data, stcs_a[0].vertices,
                               stcs_a[0].tmin, stcs_a[0].tstep,
                               stcs_a[0].subject)


vertno_max, time_max = average_stc_a.get_peak(hemi='rh')

surfer_kwargs = dict(
    hemi='split', subjects_dir=subjects_dir,
    colormap='magma',
    colorbar=False,
    background='white',
    foreground='black',
    clim=dict(kind='value', lims=[3, 4, 5]),
    views='par',
    initial_time=time_max, time_unit='s',
    size=(1600, 1000), smoothing_steps=10,
    show_traces=False)
brain = average_stc_a.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='black',
               scale_factor=0.5, alpha=0.75)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)
screenshot = brain.screenshot()
brain.close()

nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
axes.imshow(cropped_screenshot)
axes.axis('off')
# add a vertical colorbar with the same properties as the 3D one
divider = make_axes_locatable(axes)
cax = divider.append_axes('right', size='5%', pad=0.2)
cbar = mne.viz.plot_brain_colorbar(cax, clim=dict(kind='value', lims=[3, 4, 5]),
                                   colormap='magma',
                                   label='Activation (F)')
fig.subplots_adjust(
    left=0.15, right=0.9, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)
fig_name = fname.figures + '/STC_A_170ms.pdf'
fig.savefig(fig_name, dpi=600)



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