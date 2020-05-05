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

from mne import read_epochs, grand_average
from mne.viz import plot_compare_evokeds

# All parameters are defined in config.py
from config import subjects, fname, LoggingFormat
from stats import within_subject_cis

a_cues = dict()
b_cues = dict()

a_erps = dict()
b_erps = dict()

baseline = (-0.300, -0.050)

###############################################################################
# 1) loop through subjects and compute ERPs for A and B cues
for subj in subjects:

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
             't6': [2.0, 0.50]}

# plot GFP and save figure
evokeds = {'A Cue': ga_a_cue.copy().crop(tmin=-0.25),
           'B Cue': ga_b_cue.copy().crop(tmin=-0.25)}

# use viridis colors
colors = np.linspace(0, 1, len(gfp_times.values()))
# create plot
fig, ax = plt.subplots(figsize=(8, 3))
plot_compare_evokeds(evokeds,
                     axes=ax,
                     linestyles={'A Cue': '-', 'B Cue': '--'},
                     styles={'A Cue': {"linewidth": 2.0},
                             'B Cue': {"linewidth": 2.0}},
                     legend='upper center',
                     ylim=dict(eeg=[-0.1, 4]),
                     colors={'A Cue': 'k', 'B Cue': 'crimson'})
ax.set_xticks(list(np.arange(-.25, 2.55, 0.25)), minor=False)
ax.set_yticks(list(np.arange(0, 5, 1)), minor=False)
# annotate thr gpf plot
for i, val in enumerate(gfp_times.values()):
    ax.bar(val[0], 5, width=val[1], alpha=0.20,
           align ='edge', color=cm.viridis(colors[i]))
ax.annotate('t1', xy=(0.075, 4.), weight="bold")
ax.annotate('t2', xy=(0.155, 4.), weight="bold")
ax.annotate('t3', xy=(0.27, 4.), weight="bold")
ax.annotate('t4', xy=(0.45, 4.), weight="bold")
ax.annotate('t5', xy=(0.63, 4.), weight="bold")
ax.annotate('t6', xy=(2.24, 4.), weight="bold")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 4)
ax.spines['bottom'].set_bounds(-0.25, 2.5)
ax.xaxis.set_label_coords(0.5, -0.13)
fig.subplots_adjust(bottom=0.15)

# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-10, 10]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [.100, .180, .310, .500, .680, 1.00, 2.45]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='s',
                    vmin=8, vmax=-8,
                    average=0.05,
                    outlines='skirt')

ga_a_cue.plot_joint(ttp, ts_args=ts_args, topomap_args=topomap_args)
ga_b_cue.plot_joint(ttp, ts_args=ts_args, topomap_args=topomap_args)

cis = within_subject_cis([a_erps, b_erps])
electrode = 'PO8'
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




ga_b_cue.plot_image(xlim=[-0.25, 2.5], clim=dict(eeg=[-8, 8]))
ga_a_cue.plot_image(xlim=[-0.25, 2.5], clim=dict(eeg=[-8, 8]))

