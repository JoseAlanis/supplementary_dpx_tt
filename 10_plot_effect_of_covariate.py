"""
=============================================================
Compute test statistics for effect of condition and moderator
=============================================================

Compute T- and F- for the effect of conditions and search for
significant spatio-temporal clusters of activity. Further,
estimate the moderating effect of behavioral performance measures
on a group-level.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np
import pandas as pd

import matplotlib.pylab as plt

from scipy.stats import zscore

from sklearn.linear_model import LinearRegression

from mne.decoding import get_coef
from mne.viz import plot_compare_evokeds
from mne import read_epochs, EvokedArray

from config import subjects, fname
# exclude subjects 51
subjects = subjects[subjects != 51]

# load individual beta coefficients (effect of condition)
betas = np.load(fname.results + '/subj_betas_cue_m250.npy')
# load bootstrap betas (effect of moderator)
betas_pbi = np.load(fname.results + '/pbi_rt_betas_m250.npy')
# load model R-squared
r2 = np.load(fname.results + '/subj_r2_cue_m250.npy')

# information about subjects' performance in the task
pbi_rt = pd.read_csv(fname.results + '/pbi.tsv', sep='\t', header=0)

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
channels = cue_epo.ch_names
n_channels = len(epochs_info['ch_names'])
n_times = len(cue_epo.times)
times = cue_epo.times
tmin = cue_epo.tmin

###############################################################################
# 2) create group-level design matrix for effect of moderator

# z-score PBI predictor
pbi_rt = pbi_rt.drop('subject', axis=1)
pbi_rt['pbi_rt_z'] = zscore(pbi_rt.pbi_rt)

# create group-level design matrix
pbi_rt = pbi_rt.assign(intercept=1)
group_design = pbi_rt[['intercept', 'pbi_rt_z']]

###############################################################################
# 3) set up and fit model for effect of moderator (i.e., PBI)r

linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(group_design, betas)

# extract group-level beta coefficients
group_coefs = get_coef(linear_model, 'coef_')

# create evoked object containing the estimated effect of moderator
pred_i = np.where(group_design.columns == 'pbi_rt_z')[0]
# store regression coefficient for moderator (i.e., PBI)
group_betas = group_coefs[:, pred_i]
# back projection to channels x time points
group_betas = group_betas.reshape((n_channels, n_times))
# create evoked object containing the back projected coefficients
group_betas_evoked = EvokedArray(group_betas, epochs_info, tmin)

###############################################################################
# 4) plot effect of moderator (i.e, model estimates)
# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-2, 2]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [0.45, 0.55, 0.65, 0.75, 2.35]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='ms',
                    vmin=1.2, vmax=-1.2,
                    average=0.05,
                    extrapolate='head')

title = 'Moderating effect of PBI (64 EEG channels)'
fig = group_betas_evoked.plot_joint(times=ttp,
                                    ts_args=ts_args,
                                    topomap_args=topomap_args,
                                    title=title)
fig.axes[-1].texts[0]._fontproperties._size = 12.0  # noqa
fig.axes[-1].texts[0]._fontproperties._weight = 'bold'  # noqa
fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig.axes[0].set_xticklabels(list(np.arange(-250, 2550, 250)))
fig.axes[0].set_xlabel('Time (ms)')
fig.axes[0].set_yticks(list(np.arange(-1.5, 1.55, 1.0)), minor=False)
fig.axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axvline(x=0, ymin=-2, ymax=2,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(-1.5, 1.5)
fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)
fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1.15, h * 1.15)
fig_name = fname.figures + '/PBI_rt_betas_%s.pdf'
fig.savefig(fig_name, dpi=300)


# compute CI boundaries according to:
# Pernet, C. R., Chauveau, N., Gaspar, C., & Rousselet, G. A. (2011).
# LIMO EEG: a toolbox for hierarchical LInear MOdeling of
# ElectroEncephaloGraphic data.
# Computational intelligence and neuroscience, 2011, 3.
# i.e., CI = ß(a+1), ß(c)
# a = (alpha * number of bootstraps) / (2 * number of predictors)
# c = number of bootstraps - a

n_boot = betas_pbi.shape[0]
a = (0.05 * n_boot) / (2 * 1)
# c = number of bootstraps - a
c = n_boot - a

# compute low and high percentiles for bootstrapped beta coefficients
lower_b, upper_b = np.quantile(betas_pbi, [(a+1)/n_boot, c/n_boot], axis=0)

# reshape to channels * time-points space
lower_b = lower_b.reshape((n_channels, n_times))
upper_b = upper_b.reshape((n_channels, n_times))

# create plot for effect of moderator
for elec in ['Fp1', 'AFz', 'F6', 'C3', 'CPz','Pz', 'Oz', 'CP1', 'PO8', 'PO7']:
    # index of Pz in channels array
    electrode = elec
    pick = group_betas_evoked.ch_names.index(electrode)

    # create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = plot_compare_evokeds({r'Effect of $PBI_{rt}$':
                                   group_betas_evoked},
                              legend='upper center',
                              ylim=dict(eeg=[-2.5, 2.5]),
                              picks=pick,
                              show_sensors='upper right',
                              axes=ax,
                              colors=['k'],
                              show=False)
    ax[0].axes[0].fill_between(times,
                               # transform values to microvolt
                               upper_b[pick] * 1e6,
                               lower_b[pick] * 1e6,
                               alpha=0.2,
                               color='k')
    ax[0].axes[0].set_ylabel(r'$\beta$ ($\mu$V)')
    ax[0].axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                          color='black', linestyle='dashed', linewidth=.8)
    ax[0].axes[0].spines['top'].set_visible(False)
    ax[0].axes[0].spines['right'].set_visible(False)
    ax[0].axes[0].spines['left'].set_bounds(-2.0, 2.0)
    ax[0].axes[0].spines['bottom'].set_bounds(-.25, 2.5)
    ax[0].axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    ax[0].axes[0].set_xticklabels(list(np.arange(-250, 2550, 250)))
    ax[0].axes[0].set_xlabel('Time (ms)')
    for t in times[(lower_b[pick] * 1e6 > 0.05) | (upper_b[pick] * 1e6 <
                                                   -0.05)]:
        ax[0].axes[0].scatter(t, -1.5, marker='d', color='darkmagenta', s=25.0)
    plt.plot()
    fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 1.15, h * 1.15)
    fig_name = fname.figures + '/PBI_rt_betas_%s.pdf' % elec
    fig.savefig(fig_name, dpi=300)
