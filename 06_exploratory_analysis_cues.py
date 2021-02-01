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
font = 'Arial'  # noqa

from mne import read_epochs, combine_evoked, grand_average
from mne.channels import make_1020_channel_selections
from mne.viz import plot_compare_evokeds

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
plt.hist(lat_a, 10, alpha=0.5, label='Cue A')
plt.hist(lat_b, 10, alpha=0.5, label='Cue B')
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
fig, ax = plt.subplots(figsize=(7, 2.5))
plot_compare_evokeds(evokeds,
                     axes=ax,
                     linestyles={'Cue A': '-', 'Cue B': '--'},
                     styles={'Cue A': {"linewidth": 1.5},
                             'Cue B': {"linewidth": 1.5}},
                     ylim=dict(eeg=[-0.1, 4.0]),
                     colors={'Cue A': 'k', 'Cue B': 'crimson'},
                     show=False)
ax.set_title('A) Cue evoked GFP', size=16.0, pad=20.0, loc='left',
             fontweight='bold', fontname=font)
ax.set_xticks(list(np.arange(-.25, 2.55, 0.25)), minor=False)
ax.set_xticklabels(list(np.arange(-250, 2550, 250)), fontname=font)
ax.set_xlabel('Time (ms)', labelpad=10.0)
ax.set_ylabel(r'$\mu$V', labelpad=10.0)
ax.set_yticks(list(np.arange(0, 5, 1)), minor=False)
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
ax.legend(loc='upper right', framealpha=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 4)
ax.spines['bottom'].set_bounds(-0.25, 2.5)
fig.subplots_adjust(bottom=0.20, top=0.80)
fig.savefig(fname.figures + '/GFP_evoked_cues.pdf', dpi=300)

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

    fig.axes[-1].texts[0]._fontproperties._size=12.0
    fig.axes[-1].texts[0]._fontproperties._weight='bold'
    fig.axes[0].tick_params(axis='both', which='major', labelsize=12)
    fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    fig.axes[0].set_xticklabels(list(np.arange(-250, 2550, 250)),
                                fontname=font)
    fig.axes[0].set_xlabel('Time (ms)', fontname=font)
    fig.axes[0].set_yticks(list(np.arange(-8, 8.5, 4)), minor=False)
    fig.axes[0].set_yticklabels(list(np.arange(-8, 8.5, 4)),
                                fontname=font)
    fig.axes[0].set_xlabel('Time (ms)', labelpad=10.0, size=12)
    fig.axes[0].set_ylabel(r'$\mu$V', labelpad=10.0, size=12)
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
    fig_name = fname.figures + '/Evoked_%s.pdf' % evoked.replace(' ', '_')
    fig.savefig(fig_name, dpi=300)

###############################################################################
# 6) plot difference wave (Cue B - Cue A)

# compute difference wave
ab_diff = combine_evoked([ga_b_cue, -ga_a_cue], weights='equal')

# make channel ROIs for easier interpretation of the plot
selections = make_1020_channel_selections(ga_a_cue.info, midline='12z')

# get colormap and create figure
colormap = cm.get_cmap('RdBu_r')
fig = plt.figure(figsize=(23.0, 6.0))
axes = [plt.subplot2grid((6, 20), (0, 0), rowspan=6, colspan=5),
        plt.subplot2grid((6, 20), (0, 6), rowspan=6, colspan=5),
        plt.subplot2grid((6, 20), (0, 12), rowspan=6, colspan=5),
        plt.subplot2grid((6, 20), (1, 18), rowspan=2, colspan=1)]
for s, selection in enumerate(selections):
    picks = selections[selection]

    mask = abs(ab_diff.data) > 1e-6

    ab_diff.plot_image(xlim=[-0.25, 2.5],
                       picks=picks,
                       clim=dict(eeg=[-3, 3]),
                       colorbar=False,
                       axes=axes[s],
                       mask=mask,
                       mask_cmap='RdBu_r',
                       mask_alpha=0.5,
                       show=False)
    # tweak plot appearance
    if selection in {'Left', 'Right'}:
        title = selection + ' hemisphere'
    else:
        title = 'Midline'
    axes[s].set_title(title, size=16.0, pad=10.0, loc='center',
                      fontweight='bold', fontname=font)
    axes[s].set_ylabel('EEG sensors', labelpad=10.0, fontsize=12.0)
    axes[s].set_xlabel('Time (s)', labelpad=10.0, fontsize=12.0)

    axes[s].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    axes[s].set_xticklabels(list(np.arange(-250, 2550, 250)),
                            rotation=40)
    axes[s].set_yticks(np.arange(len(picks)), minor=False)
    labels = [ga_a_cue.ch_names[i] for i in picks]
    axes[s].set_yticklabels(labels, minor=False)
    axes[s].spines['top'].set_visible(False)
    axes[s].spines['right'].set_visible(False)
    axes[s].spines['left'].set_bounds(-0.5, len(picks)-0.5)
    axes[s].spines['bottom'].set_bounds(-.25, 2.5)
    axes[s].texts = []

    # add intercept line (at 0 s) and customise figure boundaries
    axes[s].axvline(x=0, ymin=0, ymax=len(picks),
                    color='black', linestyle='dashed', linewidth=1.0)

    # adjust margins
    plt.subplots_adjust(bottom=0.15, wspace=0.5, hspace=0.25)

# add colorbar
orientation = 'vertical'
norm = Normalize(vmin=-3.0, vmax=3.0)
cbar = ColorbarBase(axes[-1], cmap=colormap,
                    ticks=[-3.0, -1.5, 0., 1.5, 3.0], norm=norm,
                    label=r'Difference B-A ($\mu$V)',
                    orientation=orientation)
cbar.outline.set_visible(False)
cbar.ax.set_frame_on(True)
label = r'Difference B-A (in $\mu$V)'
for key in ('left', 'top',
            'bottom' if orientation == 'vertical' else 'right'):
    cbar.ax.spines[key].set_visible(False)

# save figure
fig.savefig(fname.figures + '/Diff_A-B_image.pdf', dpi=300)

# ** plot topography of the difference wave **
# variables for plot
ttp = [0.20, 0.30, 0.60, 0.80, 1.0, 1.5, 2.30]
# create plot
fig = plt.figure(figsize=(15, 2.0))
axes = [plt.subplot2grid((6, 23), (0, 0), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 3), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 6), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 9), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 12), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 15), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (0, 18), rowspan=6, colspan=3),
        plt.subplot2grid((6, 23), (1, 22), rowspan=4, colspan=1)]
for ti, t in enumerate(ttp):
    ab_diff.plot_topomap(times=t,
                         average=0.05,
                         vmin=-3.0, vmax=3.0,
                         extrapolate='head',
                         colorbar=False,
                         axes=axes[ti],
                         show=False)
orientation = 'vertical'
norm = Normalize(vmin=-3.0, vmax=3.0)
cbar = ColorbarBase(axes[-1], cmap=colormap,
                    ticks=[-3.0, -1.5, 0., 1.5, 3.0], norm=norm,
                    label=r'Difference B-A ($\mu$V)',
                    orientation=orientation)
cbar.outline.set_visible(False)
cbar.ax.set_frame_on(True)
label = r'Difference B-A (in $\mu$V)'
for key in ('left', 'top',
            'bottom' if orientation == 'vertical' else 'right'):
    cbar.ax.spines[key].set_visible(False)
fig.savefig(fname.figures + '/Diff_Topomaps.pdf', dpi=300)

###############################################################################
# 7) Plot ERPs for individual electrodes of interest
cis = within_subject_cis([a_erps, b_erps])

for electrode in ['FCz', 'FC1', 'FC3', 'Cz', 'C1', 'C3',
                  'Pz', 'Oz', 'PO8', 'PO7']:
    pick = ga_a_cue.ch_names.index(electrode)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_compare_evokeds({'Cue A': ga_a_cue.copy().crop(-0.25, 2.5),
                          'Cue B': ga_b_cue.copy().crop(-0.25, 2.5)},
                         vlines=[],
                         picks=pick,
                         invert_y=False,
                         ylim=dict(eeg=[-8.5, 8.5]),
                         colors={'Cue A': 'k', 'Cue B': 'crimson'},
                         axes=ax,
                         truncate_xaxis=False,
                         show_sensors='upper right',
                         show=False)
    ax.axhline(y=0, xmin=-.25, xmax=2.5,
               color='black', linestyle='dotted', linewidth=.8)
    ax.axvline(x=0, ymin=-8.5, ymax=8.5,
               color='black', linestyle='dotted', linewidth=.8)
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
    ax.legend(loc='upper left', framealpha=1)
    ax.set_xlabel('Time (s)', labelpad=10.0, fontsize=11.0)
    ax.set_ylim(-8.5, 8.5)
    ax.set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
    ax.set_yticks(list(np.arange(-8, 8.5, 2)), minor=False)
    ax.set_xticklabels([str(lab) for lab in np.arange(-.25, 2.55, .25)],
                       minor=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(-8, 8)
    ax.spines['bottom'].set_bounds(-.25, 2.5)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(fname.figures + '/ERP_AB_%s.pdf' % electrode,
                dpi=300)
