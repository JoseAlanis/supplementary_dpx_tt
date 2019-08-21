# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- fit linear model,
# --- create results figures

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir

import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import PatchCollection

import numpy as np
from scipy.stats import zscore

from mne import grand_average
from mne.viz import plot_compare_evokeds

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
from mne.channels import find_ch_connectivity
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne.viz import plot_topomap, plot_compare_evokeds, tight_layout
from mne import combine_evoked, find_layout

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if op.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# derivatives path
derivatives_path = op.join(root_path, 'derivatives')

# path to eeg files
data_path = op.join(derivatives_path, 'all_epochs')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'results')):
    mkdir(op.join(derivatives_path, 'results'))

# path for saving output
output_path = op.join(derivatives_path, 'results')

# ========================================================================
# execute this cell to load previously saved cue epochs data
with open(op.join(data_path, 'all_cue_epochs.pkl'), 'rb') as cues:
    cues_dict = pickle.load(cues)

# execute this cell to load previously saved probe epochs data
with open(op.join(data_path, 'all_probe_epochs.pkl'), 'rb') as probes:
    probes_dict = pickle.load(probes)

# clean up
del cues, probes


# --- 1) only keep correct trials -------------------------------
# apply baseline to epochs and extract relevant data
for subject in cues_dict.keys():
    # extract correct trials
    cues_dict[subject] = \
        cues_dict[subject]['Correct A', 'Correct B'].apply_baseline(
            (-.3, -.05)).crop(tmin=-.25, tmax=2.45)

# only use A-cues
# for subject in cues_dict.keys():
#     # extract correct trials
#     cues_dict[subject] = \
#         cues_dict[subject]['Correct A'].apply_baseline(
#             (-.3, -.05)).crop(tmin=-.25, tmax=2.45)


# --- 2) regression parameters ----------------------------------
# subjects
subjects = list(cues_dict.keys())

# variables to be used in the analysis (i.e., predictors)
predictors = ['intercept', 'cue a - cue b']
# predictors = ['intercept', 'set-count']

# number of predictors
n_predictors = len(predictors)

# save epochs information (needed for creating a homologous
# epochs object containing linear regression result)
epochs_info = cues_dict[str(subjects[0])].info

# number of channels and number of time points in each epoch
# we'll use this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(epochs_info['ch_names'])
n_times = len(cues_dict[str(subjects[0])].times)

# also save times first time-point in data
times = cues_dict[str(subjects[0])].times
tmin = cues_dict[str(subjects[0])].tmin

# --- 3) create empty objects  for the storage of results ------

# place holders for bootstrap samples
betas = np.zeros((len(predictors[1:]), len(cues_dict.values()),
                  n_channels * n_times))

# dicts for results evoked-objects
betas_evoked = dict()
t_evokeds = dict()
r2_evoked = dict()

# --- 4) run regression analysis for each subject ---------------

# loop through subjects, set up and fit linear model
for subj_ind, subject in enumerate(cues_dict.values()):

    # --- 1) create design matrix ---
    # use epochs metadata
    design = subject.metadata.copy()

    # add intercept (constant) to design matrix
    design = design.assign(intercept=1)

    # effect code contrast for categorical variable (i.e., condition a vs. b,
    # block 0 vs block 1)
    design['cue a - cue b'] = np.where(design['cue'] == 'A', -1, 1)
    # design['set-count'] = design['run']
    # design['set-count'] = zscore(design['run'])

    # order columns of design matrix
    design = design[predictors]

    # --- 2) vectorize (eeg-channel) data for linear regression analysis ---
    # data to be analysed
    data = subject.get_data()

    # vectorize data across channels
    Y = Vectorizer().fit_transform(data)

    # --- 3) fit linear model with sklearn's LinearRegression ---
    # we already have an intercept column in the design matrix,
    # thus we'll call LinearRegression with fit_intercept=False
    linear_model = LinearRegression(fit_intercept=False)
    linear_model.fit(design, Y)

    lm_betas = dict()
    r_squared = dict()

    # --- 4) extract the resulting coefficients (i.e., betas) ---
    # column of betas array (i.e., predictor) to run bootstrap on
    for prend_ind, predictor in enumerate(predictors[1:]):

        # get estimates for predictors
        pred_col = predictors.index(predictor)

        # extract betas
        coefs = get_coef(linear_model, 'coef_')
        # only keep relevant predictor
        betas[prend_ind, subj_ind, :] = coefs[:, pred_col]

        # extract coefficients
        beta = betas[prend_ind, subj_ind, :]
        # back projection to channels x time points
        beta = beta.reshape((n_channels, n_times))
        # create evoked object containing the back projected coefficients
        lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)

    # save results
    betas_evoked[str(subjects[subj_ind])] = lm_betas

    # compute model r-squared for subject
    r2 = r2_score(Y, linear_model.predict(design), multioutput='raw_values')
    # project r-squared back to channels by times space
    r2 = r2.reshape((n_channels, n_times))
    r_squared = EvokedArray(r2, epochs_info, tmin)
    # save r-squared
    r2_evoked[str(subjects[subj_ind])] = r_squared

    # clean up
    del linear_model


# --- 5) compute mean beta-coefficient for predictor phase-coherence
# how many predictors
betas = betas[0, :, :]

# subject ids
subjects = [str(subj) for subj in subjects]

# extract cue betas for each subject
cue_effect = [betas_evoked[subj]['cue a - cue b'] for subj in subjects]
# block_effect = [betas_evoked[subj]['set-count'] for subj in subjects]
cue_r2 = [r2_evoked[subj] for subj in subjects]

# average betas
weights = np.repeat(1 / len(subjects), len(subjects))
ga_cue_effect = combine_evoked(cue_effect, weights=weights)
# ga_block_effect = combine_evoked(block_effect, weights=weights)
# average r-squared
ga_cue_r2 = combine_evoked(cue_r2, weights=weights)

# plot r-squared
ts_args = dict(xlim=(-.25, 2.45),
               unit=False,
               scalings=dict(eeg=1),
               ylim=dict(eeg=[0, 0.06]))
topomap_args = dict(cmap='Reds', scalings=dict(eeg=1),
                    vmin=0, vmax=0.06, average=0.05)
# create plot
fig = ga_cue_r2.plot_joint(ts_args=ts_args,
                           topomap_args=topomap_args,
                           title='Proportion of variance explained by '
                                 'predictors',
                           times=[.22, .59, 1.02])
fig.axes[0].set_ylabel('R-squared')

# --- 6) compute bootstrap confidence interval
# for cue betas and t-values

# set random state for replication
random_state = 42
random = np.random.RandomState(random_state)

# number of random samples
boot = 2000

# place holders for bootstrap samples
cluster_H0 = np.zeros(boot)
f_H0 = np.zeros(boot)

# setup connectivity
n_tests = betas.shape[1]
connectivity, ch_names = find_ch_connectivity(epochs_info, ch_type='eeg')
connectivity = _setup_connectivity(connectivity, n_tests, n_times)

# threshond for clustering
threshold = 35.

# # alternative threshold such as pr(>F) [WIP]
# tail = 1
# p_thresh = 0.05 / (1 + (1 == 0))
# dfn = 1
# dfd = len(betas) - 1
# threshold = stats.f.ppf(1. - p_thresh, dfn, dfd)

# # or TFCE
# threshold = dict(start=.1, step=.1)

# run bootstrap for regression coefficients
for i in range(boot):
    # extract random subjects from overall sample
    resampled_subjects = random.choice(range(betas.shape[0]),
                                       betas.shape[0],
                                       replace=True)
    # resampled betas
    resampled_betas = betas[resampled_subjects, :]

    # compute standard error of bootstrap sample
    se = resampled_betas.std(axis=0) / np.sqrt(resampled_betas.shape[0])

    # center re-sampled betas around zero
    for subj_ind in range(resampled_betas.shape[0]):
        resampled_betas[subj_ind, :] = resampled_betas[subj_ind, :] - \
                                       betas.mean(axis=0)

    # compute t-values for bootstrap sample
    t_val = resampled_betas.mean(axis=0) / se
    # transform to f-values
    f_vals = t_val ** 2

    # transpose for clustering
    f_vals = f_vals.reshape((n_channels, n_times))
    f_vals = np.transpose(f_vals, (1, 0))
    f_vals = f_vals.ravel()

    # compute clustering on squared t-values (i.e., f-values)
    clusters, cluster_stats = _find_clusters(f_vals,
                                             threshold=threshold,
                                             connectivity=connectivity,
                                             tail=1)
    # save max cluster mass. Combined, the max cluster mass values from
    # computed on the basis of the bootstrap samples provide an approximation
    # of the cluster mass distribution under H0
    if len(clusters):
        cluster_H0[i] = cluster_stats.max()
    else:
        cluster_H0[i] = np.nan

    # save max f-value
    f_H0[i] = f_vals.max()

    print(i)

# --- 7) estimate t-test based on original cue effect betas
# estimate t-values and f-values
se = betas.std(axis=0) / np.sqrt(betas.shape[0])
t_vals = betas.mean(axis=0) / se
f_vals = t_vals ** 2

# transpose for clustering
f_vals = f_vals.reshape((n_channels, n_times))
f_vals = np.transpose(f_vals, (1, 0))
f_vals = f_vals.reshape((n_times * n_channels))

# find clusters
clusters, cluster_stats = _find_clusters(f_vals,
                                         threshold=threshold,
                                         connectivity=connectivity,
                                         tail=1)


# --- 8) compute cluster significance and get mask por plot
# here, we use the distribution of cluster mass bootstrap values
cluster_thresh = np.quantile(cluster_H0, [.99], axis=0)

# clsuers above alpha level
sig_mask = cluster_stats > cluster_thresh

# back projection to channels x time points
t_vals = t_vals.reshape((n_channels, n_times))
f_vals = np.transpose(f_vals.reshape((n_times, n_channels)), (1, 0))
sig_mask = np.transpose(sig_mask.reshape((n_times, n_channels)), (1, 0))


# --- 9) create evoked object containing the resulting t-values
group_t = dict()
group_t['phase-coherence'] = EvokedArray(t_vals, epochs_info, tmin)

# electrodes to plot (reverse order to be compatible whit LIMO paper)
picks = group_t['phase-coherence'].ch_names[::-1]
# plot t-values, masking non-significant time points.
fig = group_t['phase-coherence'].plot_image(time_unit='s',
                                            # picks=picks,
                                            mask=sig_mask,
                                            xlim=(-.1, None),
                                            unit=False,
                                            # keep values scale
                                            scalings=dict(eeg=1))
fig.axes[1].set_title('T-value')
fig.axes[0].set_title('Group-level effect of phase-coherence')
fig.set_size_inches(7, 4)


# --- 10) compute significance level for clusters
# get upper CI bound from cluster mass H0
clust_threshold = np.quantile(cluster_H0[~np.isnan(cluster_H0)], [.95])

# good cluster inds
good_cluster_inds = np.where(cluster_stats > clust_threshold)[0]

# reshape clusters
clusters = _reshape_clusters(clusters, (n_times, n_channels))

# --- 8) back projection to channels x time points
t_vals = t_vals.reshape((n_channels, n_times))
f_vals = f_vals.reshape((n_times, n_channels))


# --- 9) create evoked object containing the resulting t-values
group_t = dict()
group_t['cue a - cue b'] = EvokedArray(np.transpose(f_vals, (1, 0)),
                                       # t_vals,
                                       epochs_info,
                                       tmin)
# scaled values for plot
group_t['cue a - cue b (scaled)'] = EvokedArray(np.transpose(f_vals * 1e-6,
                                                             (1, 0)),
                                                # t_vals * 1e-6,
                                                epochs_info,
                                                tmin)

# electrodes to plot (reverse order to be compatible whit LIMO paper)
picks = group_t['cue a - cue b'].ch_names[::-1]
# plot t-values, masking non-significant time points.
fig = group_t['cue a - cue b'].plot_image(time_unit='s',
                                          # picks=picks,
                                          # mask=sig_mask,
                                          # xlim=(0., None),
                                          unit=False,
                                          # keep values scale
                                          scalings=dict(eeg=1),
                                          cmap='Reds',
                                          clim=dict(eeg=[0, None])
                                          )
fig.axes[1].set_title('F-value')
fig.axes[0].set_title('Group-level effect of Cue')
fig.set_size_inches(6.5, 4)


# --- 10) visualize clusters
# get sensor positions via layout
pos = find_layout(epochs_info).pos

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = f_vals[time_inds, :].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(f_map, pos, mask=mask, axes=ax_topo, cmap='Reds',
                            vmin=np.min, vmax=np.max, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"

    plot_compare_evokeds(ga_cue_effect,
                         title=title,
                         picks=ch_inds,
                         combine='mean',
                         axes=ax_signals,
                         show=False,
                         split_legend=True,
                         truncate_yaxis='max_ticks')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
    ax_signals.set_ylabel('effect cue a - cue b')

    # clean up viz
    tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
