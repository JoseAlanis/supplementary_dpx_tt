"""
==================================
Exploratory analysis of cue epochs
==================================

Compute descriptive statistics and exploratory analysis plots
for cue locked epochs.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne import read_epochs, combine_evoked, grand_average
from mne.channels import make_1020_channel_selections
from mne.viz import plot_compare_evokeds, plot_brain_colorbar

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat
from stats import within_subject_cis

# dicts for storing individual sets of epochs/ERPs
a_cues = dict()
b_cues = dict()
a_erps = dict()
b_erps = dict()

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
    a_cues['subj_%s' % subj] = cue_epo['Correct A'].apply_baseline(baseline)
    b_cues['subj_%s' % subj] = cue_epo['Correct B'].apply_baseline(baseline)

    # compute ERP
    a_erps['subj_%s' % subj] = a_cues['subj_%s' % subj].average()
    b_erps['subj_%s' % subj] = b_cues['subj_%s' % subj].average()

###############################################################################
# 2) compute grand averages
ga_a_cue = grand_average(list(a_erps.values()))
ga_b_cue = grand_average(list(b_erps.values()))

###############################################################################
# 3) plot global field power
gfp_times = {'t1': [0.07, 0.07],
             't2': [0.14, 0.10],
             't3': [0.24, 0.12],
             't4': [0.36, 0.24],
             't5': [0.60, 0.15],
             't6': [0.75, 0.25],
             't7': [2.0, 0.50]}

# create evokeds dict
evokeds = {'A Cue': ga_a_cue.copy().crop(tmin=-0.25),
           'B Cue': ga_b_cue.copy().crop(tmin=-0.25)}

# use viridis colors
colors = np.linspace(0, 1, len(gfp_times.values()))
cmap = cm.get_cmap('viridis')
# plot GFP and save figure
fig, ax = plt.subplots(figsize=(8, 3))
plot_compare_evokeds(evokeds,
                     axes=ax,
                     linestyles={'A Cue': '-', 'B Cue': '--'},
                     styles={'A Cue': {"linewidth": 2.0},
                             'B Cue': {"linewidth": 2.0}},
                     ylim=dict(eeg=[-0.1, 4]),
                     colors={'A Cue': 'k', 'B Cue': 'crimson'})
ax.set_xticks(list(np.arange(-.25, 2.55, 0.25)), minor=False)
ax.set_yticks(list(np.arange(0, 5, 1)), minor=False)
# annotate the gpf plot and tweak it's appearance
for i, val in enumerate(gfp_times.values()):
    ax.bar(val[0], 5, width=val[1], alpha=0.20,
           align='edge', color=cmap(colors[i]))
ax.annotate('t1', xy=(0.075, 4.), weight="bold")
ax.annotate('t2', xy=(0.155, 4.), weight="bold")
ax.annotate('t3', xy=(0.27, 4.), weight="bold")
ax.annotate('t4', xy=(0.45, 4.), weight="bold")
ax.annotate('t5', xy=(0.635, 4.), weight="bold")
ax.annotate('t6', xy=(0.845, 4.), weight="bold")
ax.annotate('t7', xy=(2.24, 4.), weight="bold")
ax.legend(loc='upper right', framealpha=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 4)
ax.spines['bottom'].set_bounds(-0.25, 2.5)
ax.xaxis.set_label_coords(0.5, -0.175)
fig.subplots_adjust(bottom=0.2)
fig.savefig(fname.figures + '/GFP_evoked_cues.pdf', dpi=300)

###############################################################################
# 4) plot condition ERPs
# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-10, 10]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [0.11, 0.18, 0.30, 0.50, 0.68, 0.90, 2.35]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='s',
                    vmin=8, vmax=-8,
                    average=0.05,
                    extrapolate='local')

fig = ga_a_cue.plot_joint(ttp,
                          ts_args=ts_args,
                          topomap_args=topomap_args,
                          title='Cue A (64 EEG channels)')
fig.axes[-1].texts[0]._fontproperties._size=12.0  # noqa
fig.axes[-1].texts[0]._fontproperties._weight='bold'  # noqa
fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig.axes[0].set_yticks(list(np.arange(-8, 8.5, 4)), minor=False)
fig.axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axvline(x=0, ymin=-5, ymax=5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(-8, 8)
fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)
fig.axes[0].xaxis.set_label_coords(0.5, -0.2)
w, h = fig.get_size_inches()
fig.set_size_inches(w * 1.15, h * 1.15)
fig.savefig(fname.figures + '/Evoked_A_Cue.pdf', dpi=300)

ga_b_cue.plot_joint(ttp, ts_args=ts_args, topomap_args=topomap_args)





# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# ga_a_cue.plot_image(xlim=[-0.25, 2.5],
#                     clim=dict(eeg=[-8, 8]),
#                     axes=ax[0])
# ga_b_cue.plot_image(xlim=[-0.25, 2.5],
#                     clim=dict(eeg=[-8, 8]),
#                     axes=ax[1])

ab_diff = combine_evoked([ga_b_cue, -ga_a_cue], weights='equal')

selections = make_1020_channel_selections(ga_a_cue.info, midline='12z')

frontal = [i for i in ga_b_cue.ch_names if i.startswith('F') or i.startswith('A')]
central = [i for i in ga_b_cue.ch_names if i.startswith('C')]
parietal = [i for i in ga_b_cue.ch_names if i.startswith('P|O')]



frontal = ['FT7', 'F7', 'AF7', 'F5', 'F3', 'AF3', 'F1', 'AF3',
           'Fp1', 'Fpz', 'Fp2',
           'AF4', 'F2', 'AF4', 'F4', 'F6', 'AF8', 'F8', 'FT8']


x =  ['Fp1', 'Fpz', 'Fp2',
      'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
      'FT7','F7', 'F5',  'F3', 'F1',  'Fz', 'F2', 'F4', 'F6', 'F8',  'FT8']

central = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
           'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
           'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']

picks = selections
picks = selections['Midline']
picks = np.arange(0, 64)

fig, ax = plt.subplots(figsize=(8, 4))
ab_diff.plot_image(xlim=[-0.25, 2.5],
                   picks=picks,
                   clim=dict(eeg=[-5, 5]),
                   colorbar=True,
                   axes=ax)
ax.set_yticks(np.arange(len(picks)), minor=False)
labels = [ga_a_cue.ch_names[i] for i in picks]
ax.set_yticklabels(labels, minor=False)

ax.texts = []
colormap = 'RdBu_r'
clim = dict(kind='value', lims=[-5, 0, 5])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3.5%', pad=0.5)
cbar = plot_brain_colorbar(cax, clim, colormap, label=r'Difference in $\mu V$')
fig.subplots_adjust(
    left=0.1, right=0.9, wspace=0.1, hspace=0.5)
fig.savefig(fname.figures + '/Diff_A-B_image.pdf', dpi=300)

ab_diff.plot_joint()


ab_diff.plot_topomap(times='peaks',
                     average=0.05, extrapolate='local',
                     outlines='skirt')


ab_diff.plot_topomap(times=[0.2, 0.35, 0.61, 1.15, 1.55, 1.80, 2.35],
                     average=0.05, extrapolate='local',
                     outlines='skirt')

ab_diff.plot_joint()




cis = within_subject_cis([a_erps, b_erps])
electrode = 'FCz'
pick = ga_a_cue.ch_names.index(electrode)

fig, ax = plt.subplots(figsize=(8, 5))
plot_compare_evokeds({'Cue A': ga_a_cue.copy().crop(-0.5, 2.5),
                      'Cue B': ga_b_cue.copy().crop(-0.5, 2.5)},
                     picks=pick,
                     invert_y=True,
                     ylim=dict(eeg=[-7, 7]),
                     colors={'Cue A': 'k', 'Cue B': 'crimson'},
                     axes=ax,
                     truncate_xaxis=True,
                     show_sensors='lower right')
ax.fill_between(ga_a_cue.times,
                (ga_a_cue.data[pick] + cis[0, pick, :]) * 1e6,
                (ga_a_cue.data[pick] - cis[0, pick, :]) * 1e6,
                alpha=0.2,
                color='k')
ax.fill_between(ga_b_cue.times,
                (ga_b_cue.data[pick] + cis[1, pick, :]) * 1e6,
                (ga_b_cue.data[pick] - cis[1, pick, :]) * 1e6,
                alpha=0.2,
                color='crimson')
ax.set_xticks(list(np.arange(-.50, 2.55, .50)), minor=False)
ax.set_yticks(list(np.arange(-6, 6.5, 2)), minor=False)
ax.set_xticklabels([str(lab) for lab in np.arange(-.50, 2.55, .50)],
                   minor=False)
fig.axes[0].spines['bottom'].set_bounds(-0.5, 2.5)
fig.axes[0].spines['left'].set_bounds(-6, 6)
