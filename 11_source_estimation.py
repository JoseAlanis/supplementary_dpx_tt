
import os.path as op

import numpy as np

from scipy import stats as stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne.datasets import fetch_fsaverage
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       permutation_cluster_test,
                       summarize_clusters_stc)
from mne.viz import plot_brain_colorbar
from mne import read_epochs, make_forward_solution, compute_covariance, \
    SourceEstimate, spatial_src_connectivity, read_source_spaces

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
    a_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=1.5)
    b_epo = cue_epo['Correct B']
    b_epo.apply_baseline(baseline=baseline).crop(tmin=-0.3, tmax=1.5)

    a_epochs_info = a_epo.info
    b_epochs_info = b_epo.info

    equalize_epoch_counts([a_epo, b_epo])

    noise_cov_a = compute_covariance(
        a_epo, tmax=-0.05,
        method=['shrunk', 'empirical'], rank=None, verbose=True)
    noise_cov_b = compute_covariance(
        b_epo, tmax=-0.05,
        method=['shrunk', 'empirical'], rank=None, verbose=True)

    evoked_a = a_epo.average()
    evoked_b = b_epo.average()
    # evoked_a.plot(time_unit='s')
    # evoked_a.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg',
    #                       time_unit='s')
    # Show whitening:
    # evoked_a.plot_white(noise_cov_a, time_unit='s')

    fwd_a = make_forward_solution(evoked_a.info, trans=trans, src=src,
                                  bem=bem, eeg=True, mindist=5.0, n_jobs=n_jobs)
    fwd_b = make_forward_solution(evoked_b.info, trans=trans, src=src,
                                  bem=bem, eeg=True, mindist=5.0, n_jobs=n_jobs)

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

data = np.average([s.copy().crop(tmin=0.1, tmax=0.2).data for s in stcs_b],
                  axis=0)

vertices = stcs_b[0].copy().crop(tmin=0.1, tmax=0.2).vertices
tmin = stcs_b[0].copy().crop(tmin=0.1, tmax=0.2).tmin
tstep = stcs_b[0].copy().crop(tmin=0.1, tmax=0.2).tstep
subject = stcs_b[0].copy().crop(tmin=0.1, tmax=0.2).subject

average_stc_a = SourceEstimate(data,
                               vertices,
                               tmin,
                               tstep,
                               subject)


vertno_max, time_max = average_stc_a.get_peak('rh')
surfer_kwargs = dict(
    hemi='split',
    surface='pial',
    subjects_dir=subjects_dir,
    colormap='magma',
    colorbar=False,
    background='white',
    foreground='black',
    clim=dict(kind='value', lims=[3, 4.5, 6]),
    views='cau',
    initial_time=time_max,
    time_unit='s',
    size=(2500, 1000),
    smoothing_steps=10,
    show_traces=False)
brain = average_stc_a.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='black',
               scale_factor=0.5, alpha=0.75)
# brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
#                font_size=14)
screenshot = brain.screenshot()
brain.close()

nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))
axes.imshow(cropped_screenshot)
axes.axis('off')
# add a vertical colorbar with the same properties as the 3D one
divider = make_axes_locatable(axes)
cax = divider.append_axes('right', size='5%', pad=0.2)
cbar = plot_brain_colorbar(cax, clim=dict(kind='value', lims=[3, 4.5, 6]),
                           colormap='magma',
                           label='Activation (F)')
fig.subplots_adjust(
    left=0.15, right=0.9, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)
fig_name = fname.figures + '/STC_B_170ms.pdf'
fig.savefig(fig_name, dpi=600)


x = stcs_a[0].copy().crop(tmin=0.0, tmax=0.21)
tmin = x.tmin
times = x.times
tstep = x.tstep * 1000  #

n_vertices, n_times = x.data.shape
X = np.zeros((2, len(subjects), n_vertices, n_times))

for i in range(len(stcs_a)):
    print(i)
    X[0, i, ...] = stcs_b[i].copy().crop(tmin=0.0, tmax=0.21).data
    X[1, i, ...] = stcs_a[i].copy().crop(tmin=0.0, tmax=0.21).data

X = np.abs(X)  # only magnitude
X = X[0, :, :, :] - X[1, :, :, :]  # make paired contrast
#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [0, 2, 1])

Xb = X[0, :, :, :]
Xa = X[1, :, :, :]
Xb = np.transpose(Xb, [0, 2, 1])
Xa = np.transpose(Xa, [0, 2, 1])


source = read_source_spaces(src)
connectivity = spatial_src_connectivity(source)





sample_vertices = [s['vertno'] for s in inverse_operator_a['src']]

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjects) - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X,
                                       connectivity=connectivity, n_jobs=8,
                                       threshold=5.0, buffer_size=None,
                                       verbose=True)


p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjects) - 1)
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([Xb, Xa],
                             n_permutations=1000,
                             n_jobs=8,
                             threshold=t_threshold,
                             tail=0)

stc_average_a = stcs_a[0]
stc_average_b = stcs_b[0]

stc_average_a.data = np.average([s.copy().data for s in stcs_a], axis=0)
stc_average_b.data = np.average([s.copy().data for s in stcs_b], axis=0)

difference_stc = stc_average_b - stc_average_a
difference_stc.crop(tmin=0.1, tmax=0.2)


n_sources = T_obs.shape[0]
cluster_p_threshold = 0.05
indices = np.where(cluster_p_values <= cluster_p_threshold)[0]
sig_clusters = []
for index in indices:
    sig_clusters.append(clusters[index])

cluster_T = np.zeros(n_sources)
for sig_cluster in sig_clusters:
    # start = sig_cluster[0].start
    # stop = sig_cluster[0].stop
    sig_indices = np.unique(np.where(sig_cluster == 1)[0])
    cluster_T[sig_indices] = 1

t_mask = np.copy(T_obs)
t_mask[cluster_T == 0] = 0
cutoff = stats.t.ppf(1 - p_threshold / 2, df=len(subjects) - 1)

difference_stc.data[:, 1] = t_mask[1, :]

clim = dict(kind='value', lims=[cutoff, 2 * cutoff, 4 * cutoff])
brain = difference_stc.plot(subject='fsaverage',
                            subjects_dir=subjects_dir,
                            time_viewer=False, hemi='both',
                            figure=0,
                            clim=clim,
                            views='dorsal')
brain.set_time(0.61)




#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=sample_vertices,
                                             subject='fsaverage')
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='both', views='lateral', subjects_dir=subjects_dir,
    time_label='temporal extent (ms)', size=(800, 800),
    smoothing_steps=5,
    clim=dict(kind='value', pos_lims=[0, 1, 20]))
# brain.save_image('clusters.png')