"""
==================================
Exploratory analysis of cue epochs
==================================

Compute descriptive statistics and exploratory analysis plots
for cue locked ERPs.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

from scipy.stats import ttest_rel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
font = 'Mukta'  # noqa

from mne import read_epochs, combine_evoked, grand_average
from mne.viz import plot_compare_evokeds
from mne.viz.utils import _connection_line

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat
from stats import within_subject_cis

# exclude subjects 51
subjects = subjects[subjects != 51]

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
# 2) compare latency of peaks
lat_a = []
lat_b = []

# find peaks
for subj in subjects:
    _, la = a_erps['subj_%s' % subj].get_peak(tmin=0.10,
                                              tmax=0.25,
                                              mode='neg')
    lat_a.append(la)

    _, lb = b_erps['subj_%s' % subj].get_peak(tmin=0.10,
                                              tmax=0.25,
                                              mode='neg')
    lat_b.append(lb)


# plot latency effects
plt.hist(lat_a, 20, alpha=0.5, label='Cue A')
plt.hist(lat_b, 20, alpha=0.5, label='Cue B')
plt.legend(loc='upper left')
plt.savefig(fname.figures + '/N170_peak_latency.pdf', dpi=300)
plt.close()

# test for significance
ttest_rel(lat_a, lat_b)

###############################################################################
# 3) compute grand averages
ga_a_cue = grand_average(list(a_erps.values()))
ga_b_cue = grand_average(list(b_erps.values()))

###############################################################################
# 4) plot global field power
gfp_times = {'t1': [0.07, 0.07],
             't2': [0.14, 0.11],
             't3': [0.25, 0.14],
             't4': [0.39, 0.36],
             # 't5': [0.60, 0.15],
             't6': [0.90, 0.20],
             't7': [2.00, 0.45]}

# create evokeds dict
evokeds = {'Cue A': ga_a_cue.copy().crop(tmin=-0.25),
           'Cue B': ga_b_cue.copy().crop(tmin=-0.25)}

# use viridis colors
colors = np.linspace(0, 1, len(gfp_times.values()))
cmap = cm.get_cmap('viridis')
plt.rcParams.update({'mathtext.default':  'regular'})
# plot GFP and save figure
fig, ax = plt.subplots(figsize=(7, 3))
plot_compare_evokeds(evokeds,
                     axes=ax,
                     linestyles={'Cue A': '-', 'Cue B': '--'},
                     styles={'Cue A': {"linewidth": 1.5},
                             'Cue B': {"linewidth": 1.5}},
                     ylim=dict(eeg=[-0.1, 4.0]),
                     colors={'Cue A': 'k', 'Cue B': 'crimson'},
                     show=False)
ax.set_title('A) Cue evoked GFP', size=14.0, pad=20.0, loc='left',
             fontweight='bold', fontname=font)
ax.set_xlabel('Time (ms)', labelpad=10.0, font=font, fontsize=12.0)
ax.set_xticks(list(np.arange(-.25, 2.55, 0.25)), minor=False)
ax.set_xticklabels(list(np.arange(-250, 2550, 250)), fontname=font)
ax.set_ylabel(r'$\mu$V', labelpad=10.0, font=font, fontsize=12.0)
ax.set_yticks(list(np.arange(0, 5, 1)), minor=False)
ax.set_yticklabels(list(np.arange(0, 5, 1)), fontname=font)
# annotate the gpf plot and tweak it's appearance
for i, val in enumerate(gfp_times.values()):
    ax.bar(val[0], 3.9, width=val[1], alpha=0.30,
           align='edge', color=cmap(colors[i]))
ax.annotate('t1', xy=(0.070, 4.), weight="bold", fontname=font)
ax.annotate('t2', xy=(0.155, 4.), weight="bold", fontname=font)
ax.annotate('t3', xy=(0.295, 4.), weight="bold", fontname=font)
ax.annotate('t4', xy=(0.540, 4.), weight="bold", fontname=font)
# ax.annotate('t5', xy=(0.635, 4.), weight="bold")
ax.annotate('t5', xy=(0.975, 4.), weight="bold", fontname=font)
ax.annotate('t6', xy=(2.220, 4.), weight="bold", fontname=font)
ax.legend(loc='upper right', framealpha=1, prop={"family": font})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 4)
ax.spines['bottom'].set_bounds(-0.25, 2.5)
fig.subplots_adjust(bottom=0.20, top=0.80)
fig.savefig(fname.figures + '/GFP_evoked_cues.pdf', dpi=300)
plt.close('all')

###############################################################################
# 5) plot condition ERPs
# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-10, 10]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [0.11, 0.18, 0.30, 0.50, 0.68, 0.90, 2.35]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='ms',
                    vmin=8, vmax=-8,
                    average=0.05,
                    extrapolate='head',
                    outlines='head')

# plot activity pattern evoked by the cues
for evoked in evokeds:
    title = evoked.replace("_", " ") + ' (64 EEG channels)'
    fig = evokeds[evoked].plot_joint(ttp,
                                     ts_args=ts_args,
                                     topomap_args=topomap_args,
                                     title=title,
                                     show=False)

    fig.axes[-1].texts[0]._fontproperties._size = 14.0  # noqa
    fig.axes[-1].texts[0]._fontproperties._weight = 'bold' # noqa
    fig.axes[-1].texts[0]._fontproperties._family = font # noqa

    for nt, tt in enumerate(ttp):
        ms = int(tt * 1000)
        fig.axes[nt+1].set_title('%s $_{ms}$' % ms, fontname=font)

    fig.axes[0].tick_params(axis='both', which='major', labelsize=12)

    fig.axes[0].set_xlabel('Time (ms)', labelpad=10.0, font=font, fontsize=14.0)
    fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    fig.axes[0].set_xticklabels(list(np.arange(-250, 2550, 250)), fontname=font)

    fig.axes[0].set_ylabel(r'$\mu$V',  labelpad=10.0, font=font, fontsize=14.0)
    fig.axes[0].set_yticks(list(np.arange(-8, 8.5, 4)), minor=False)
    fig.axes[0].set_yticklabels(list(np.arange(-8, 8.5, 4)), fontname=font)

    fig.axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                        color='black', linestyle='dashed', linewidth=.8)
    fig.axes[0].axvline(x=0, ymin=-5, ymax=5,
                        color='black', linestyle='dashed', linewidth=.8)

    fig.axes[0].spines['top'].set_visible(False)
    fig.axes[0].spines['right'].set_visible(False)
    fig.axes[0].spines['left'].set_bounds(-8, 8)
    fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)

    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 1.0, h * 1.0)

    fig_name = fname.figures + '/Evoked_%s.pdf' % evoked.replace(' ', '_')
    fig.savefig(fig_name, dpi=300)
    plt.close('all')

###############################################################################
# 6) plot difference wave (Cue B - Cue A)

# compute difference wave
ab_diff = combine_evoked([ga_b_cue, -ga_a_cue], weights='equal')
# mask differences that are below 0.5 micro volt
mask = abs(ab_diff.data) >= 1.0e-6

# spatially defined rois for plot
rois = {
    'Frontal':
        ['F8', 'F6', 'F4', 'F2', 'Fz',
         'AF8', 'AF4', 'AFz',
         'Fp2', 'Fpz', 'Fp1',
         'AF3', 'AF7',
         'F1', 'F3', 'F5', 'F7'],
    'Central':
        ['C6',  'C4', 'C2', 'Cz',
         'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
         'FC1',  'FC3', 'FC5', 'FT7',
         'C1',  'C3', 'C5',
         ],
    'Parietal':
        ['P8', 'P6', 'P4', 'P2', 'Pz',
         'CP6',  'CP4',  'CP2', 'CPz',
         'CP1',  'CP3', 'CP5',
         'P1', 'P3', 'P5', 'P7',
         ],
    'Occipital':
        ['P10', 'PO8', 'PO4',
         'O2', 'Oz', 'O1',
         'PO3', 'PO7', 'P9']
}

# get colormap and create figure
colormap = cm.get_cmap('RdBu_r')

# create figure grid
fig = plt.figure(figsize=(7, 13))
axes = [plt.subplot2grid((40, 9), (0, 0), rowspan=8, colspan=5),
        plt.subplot2grid((40, 9), (11, 0), rowspan=8, colspan=5),
        plt.subplot2grid((40, 9), (22, 0), rowspan=8, colspan=5),
        plt.subplot2grid((40, 9), (33, 0), rowspan=6, colspan=5),

        plt.subplot2grid((40, 9), (0, 5), rowspan=6, colspan=4),
        plt.subplot2grid((40, 9), (11, 5), rowspan=6, colspan=4),
        plt.subplot2grid((40, 9), (22, 5), rowspan=6, colspan=4),
        plt.subplot2grid((40, 9), (33, 5), rowspan=6, colspan=4),

        plt.subplot2grid((40, 9), (39, 6), rowspan=1, colspan=2)
        ]

for p, pick in enumerate(rois.keys()):
    ab_diff.plot_image(xlim=[-0.25, 2.5],
                       clim=dict(eeg=[-3, 3]),
                       colorbar=False,
                       mask=mask,
                       mask_cmap='RdBu_r',
                       mask_alpha=0.5,
                       show=False,
                       axes=axes[p],
                       picks=rois[pick])

    # add line marking stimulus presentation
    axes[p].axvline(x=0, ymin=0, ymax=len(rois[pick]),
                    color='black', linestyle='dashed', linewidth=1.0)

    title = ['A) %s', 'B) %s', 'C) %s', 'D) %s']
    # add title according to specific region
    axes[p].set_title(title[p] % pick, loc='left', pad=10.0,
                      size=14.0, fontweight='bold', fontname=font)
    axes[p].set_title('', loc='center', pad=10.0,
                      size=14.0, fontweight='bold', fontname=font)

    # axis labels
    axes[p].set_ylabel('EEG sensors', labelpad=10.0, fontsize=10.5)
    axes[p].set_xlabel('Time (s)', labelpad=5.0, fontsize=11.0)

    # specify tick label size
    axes[p].tick_params(axis='both', which='major', labelsize=10)
    axes[p].tick_params(axis='both', which='minor', labelsize=8)

    # add axis ticks
    axes[p].set_xticks(list(np.arange(-.250, 2.550, .250)))
    axes[p].set_xticklabels(list(np.arange(-250, 2550, 250)),
                            rotation=45, fontname=font)

    axes[p].set_yticks(np.arange(0, len(rois[pick]), 1))
    axes[p].set_yticklabels(rois[pick], fontname=font)

    axes[p].spines['top'].set_visible(False)
    axes[p].spines['right'].set_visible(False)
    axes[p].spines['left'].set_bounds(0, len(rois[pick])-1)
    axes[p].spines['left'].set_linewidth(1.5)
    axes[p].spines['bottom'].set_bounds(-0.250, 2.500)
    axes[p].spines['bottom'].set_linewidth(1.5)

    # if any additional text in fig
    for text in axes[p].texts:
        text.set_visible(False)

# plot topomaps for times of interest
ttp = [0.550, 0.600, 0.350, 0.200]
for nt, tp in enumerate(ttp):
    ab_diff.plot_topomap(times=tp,
                         mask=mask,
                         mask_params=dict(marker='o', markerfacecolor='w',
                                          markeredgecolor='k',
                                          linewidth=0, markersize=8),
                         average=0.02,
                         vmin=-3.0, vmax=3.0,
                         extrapolate='head',
                         colorbar=False,
                         axes=axes[len(rois)+nt],
                         show=False)
    axes[len(rois)+nt].set_title('%s $_{ms}$' % int((tp * 1000)),
                                 size=16, fontname=font, weight='bold')

orientation = 'horizontal'
norm = Normalize(vmin=-3.0, vmax=3.0)
cbar = ColorbarBase(axes[-1],
                    cmap=colormap,
                    ticks=[-3.0, -1.5, 0., 1.5, 3.0],
                    norm=norm,
                    orientation=orientation)
cbar.outline.set_visible(False)
cbar.ax.set_frame_on(True)
cbar.ax.tick_params(labelsize=9)
cbar.set_label(label=r'Difference B-A ($\mu$V)', font=font, size=12)
for key in ('left', 'top',
            'bottom' if orientation == 'vertical' else 'right'):
    cbar.ax.spines[key].set_visible(False)
for tk in cbar.ax.yaxis.get_ticklabels():
    tk.set_family(font)

fig.subplots_adjust(
    left=0.100, right=1.000, bottom=0.04, top=0.970,
    wspace=0.1, hspace=0.5)

# connection lines
# draw the connection lines between time series and topoplots
lines = [_connection_line(timepoint, fig, axes[ts_ax_], axes[map_ax_])
         for timepoint, ts_ax_, map_ax_ in zip(ttp, [0, 1, 2, 3], [4, 5, 6, 7])]
for line in lines:
    fig.lines.append(line)

# mark times in time series plot
for ts_ax, timepoint in enumerate(ttp):
    height = axes[ts_ax].get_ylim()[-1] + 0.5

    axes[ts_ax].bar(timepoint, height=height, width=0.05, bottom=-0.5,
                    alpha=1.0, align='center',
                    linewidth=1.5, color='None', edgecolor='k')

fig_name = fname.figures + '/Evoked_Diff_Wave.pdf'
fig.savefig(fig_name, dpi=300)
plt.close('all')

###############################################################################
# 7) Plot ERPs for individual electrodes of interest
cis = within_subject_cis([a_erps, b_erps])

fig = plt.figure(figsize=(16, 4))
axes = [plt.subplot2grid((5, 20), (0, 0), rowspan=5, colspan=6),
        plt.subplot2grid((5, 20), (0, 7), rowspan=5, colspan=6),
        plt.subplot2grid((5, 20), (0, 14), rowspan=5, colspan=6)]
for ne, electrode in enumerate(['FC1', 'Pz', 'PO8']):
    pick = ga_a_cue.ch_names.index(electrode)

    # fig, ax = plt.subplots(figsize=(5, 4))
    plot_compare_evokeds({'Cue A': ga_a_cue.copy().crop(-0.25, 2.5),
                          'Cue B': ga_b_cue.copy().crop(-0.25, 2.5)},
                         picks=pick,
                         invert_y=False,
                         ylim=dict(eeg=[-8.5, 6.5]),
                         colors={'Cue A': 'k', 'Cue B': 'crimson'},
                         axes=axes[ne],
                         truncate_xaxis=False,
                         show_sensors='upper right',
                         show=False)
    axes[ne].fill_between(ga_a_cue.times,
                          (ga_a_cue.data[pick] + cis[0, pick, :]) * 1e6,
                          (ga_a_cue.data[pick] - cis[0, pick, :]) * 1e6,
                          alpha=0.2,
                          color='k')
    axes[ne].fill_between(ga_b_cue.times,
                          (ga_b_cue.data[pick] + cis[1, pick, :]) * 1e6,
                          (ga_b_cue.data[pick] - cis[1, pick, :]) * 1e6,
                          alpha=0.2,
                          color='crimson')

    axes[ne].set_title('%s' % electrode, loc='center', pad=5.0,
                       size=14.0, fontweight='bold', fontname=font)

    axes[ne].legend(loc='lower right', framealpha=1, prop={"family": font})

    axes[ne].set_ylabel(r'$\mu$V', labelpad=10.0, font=font, fontsize=12.0)
    axes[ne].set_yticks(list(np.arange(-8, 6.5, 2)), minor=False)
    axes[ne].set_yticklabels(np.arange(-8, 6.5, 2), fontname=font)

    axes[ne].set_xlabel('Time (s)', labelpad=10.0, font=font, fontsize=12.0)
    axes[ne].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    axes[ne].set_xticklabels(list(np.arange(-250, 2550, 250)),
                             rotation=45, fontname=font, size=11.0)

    axes[ne].axhline(y=0, xmin=-.25, xmax=2.5,
                     color='black', linestyle='dotted', linewidth=.8)
    axes[ne].axvline(x=0, ymin=-8.5, ymax=6.5,
                     color='black', linestyle='dotted', linewidth=.8)

    axes[ne].spines['top'].set_visible(False)
    axes[ne].spines['right'].set_visible(False)
    axes[ne].spines['left'].set_bounds(-8, 6)
    axes[ne].spines['bottom'].set_bounds(-.25, 2.5)

fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.25,
                    wspace=0.5, hspace=0.5)
fig.savefig(fname.figures + '/ERPs_ci_AB.pdf', dpi=300)
plt.close('all')
