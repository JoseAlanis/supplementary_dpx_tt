# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- plot erp results,
# --- create figure

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
from scipy import stats

from mne import grand_average
from mne.viz import plot_compare_evokeds

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

# --- 1) compute erps by subject -------------------------------
# cue A
erps_a_cue = {subj: cues_dict[subj]['Correct A'].apply_baseline((-.3, -.05)).average()  # noqa
              for subj in cues_dict}
# cue B
erps_b_cue = {subj: cues_dict[subj]['Correct B'].apply_baseline((-.3, -.05)).average()  # noqa
              for subj in cues_dict}

subject_erp = {subj: cues_dict[subj]['Correct B', 'Correct A'].apply_baseline((-.3, -.05)).average()  # noqa
               for subj in cues_dict}

# save times
times = cues_dict['001']['Correct A'].times

# --- 2) compute grand averages --------------------------------
# cue A
Grand_Average_A = grand_average([val for val in erps_a_cue.values()])
# cue B
Grand_Average_B = grand_average([val for val in erps_b_cue.values()])

# --- 2) compute within-subjects CIs for visualisation ---------
# see Morey (2008): Confidence Intervals from Normalized Data:
# A correction to Cousineau (2005)

# correction factor for number of conditions
n_cond = 2
corr_factor = np.sqrt(n_cond / (n_cond - 1))

# place holders for normed ERPs (condition ERP - subject ERP) + grand average
norm_erp_a = []
norm_erp_b = []

# loop through subjects and normalise ERPs
for subj in cues_dict.keys():
    # subtract subject ERP from condition ERP
    erp_a_data = (erps_a_cue[subj].data.copy() - subject_erp[subj].data.copy())
    erp_a_data = erp_a_data + Grand_Average_A.data.copy()

    # add grand average
    erp_b_data = (erps_b_cue[subj].data.copy() - subject_erp[subj].data.copy())
    erp_b_data = erp_b_data + Grand_Average_B.data.copy()

    # compute norm erp
    norm_erp_a.append(erp_a_data * corr_factor)
    norm_erp_b.append(erp_b_data * corr_factor)

# list to array
norm_erp_a = np.stack(norm_erp_a)
norm_erp_b = np.stack(norm_erp_b)

# get means
ga_a = Grand_Average_A.data
ga_b = Grand_Average_B.data

# compute standard error
sem_a = stats.sem(norm_erp_a, axis=0)
sem_b = stats.sem(norm_erp_b, axis=0)

# compute confidence interval
h_a = sem_a * stats.t.ppf((1 + 0.95) / 2., len(norm_erp_a)-1)
h_b = sem_b * stats.t.ppf((1 + 0.95) / 2., len(norm_erp_a)-1)

# compute upper and lower boundaries
upper_a = ga_a + h_a
lower_a = ga_a - h_a

upper_b = ga_b + h_b
lower_b = ga_b - h_b

# --- 3) plot ERP results ---------
# arguments fot the time-series maps
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-10, 10]),
               xlim=[-.25, 2.5])

# times to plot
ttp = [.100, .180, .310, .500, .680, 2.45]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='s',
                    vmin=8, vmax=-8,
                    average=0.05)

# --- 3.1) plot evoked activity ---------
# A cues
fig = Grand_Average_A.plot_joint(times=ttp,
                                 ts_args=ts_args,
                                 topomap_args=topomap_args,
                                 title='Average Evoked Activity - Cue A')
fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig.axes[0].set_yticks(list(np.arange(-9, 9.5, 4.5)), minor=False)
fig.axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axvline(x=0, ymin=-5, ymax=5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(-9, 9)
fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)
fig.axes[0].xaxis.set_label_coords(.5, -.135)
fig.set_size_inches(9, 4.5)
fig.savefig(op.join(output_path, 'Evoked_A_Cue.pdf'), dpi=300)

# B cues
fig = Grand_Average_B.plot_joint(times=ttp,
                                 ts_args=ts_args,
                                 topomap_args=topomap_args,
                                 title='Average Evoked Activity - Cue B')
fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig.axes[0].set_yticks(list(np.arange(-9, 9.5, 4.5)), minor=False)
fig.axes[0].axhline(y=0, xmin=-.5, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axvline(x=0, ymin=-5, ymax=5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(-9, 9)
fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)
fig.axes[0].xaxis.set_label_coords(.5, -.135)
fig.set_size_inches(9, 4.5)
fig.savefig(op.join(output_path, 'Evoked_B_Cue.pdf'), dpi=300)

# --- 3.2) plot global field power ---------
gfp_times = {'t1': [[0.07, 0], 0.07],
             't2': [[0.14, 0], 0.10],
             't3': [[0.24, 0], 0.12],
             't4': [[0.36, 0], 0.24],
             't5': [[2., 0], 0.45]}
# create annotation patches
patches = []
for key, val in gfp_times.items():
    rect = mpatches.Rectangle(val[0], val[1], 4, ec="none")
    patches.append(rect)
# use viridis colors
colors = np.linspace(0, 1, len(patches))
collection = PatchCollection(patches, cmap=plt.cm.viridis, alpha=0.2)
collection.set_array(np.array(colors))
# plot GFP and save figure
evokeds = {'A Cue': Grand_Average_A.copy().crop(tmin=-.25),
           'B Cue': Grand_Average_B.copy().crop(tmin=-.25)}
# colors
cmap = 'magma'
colors = {'A Cue': 0, 'B Cue': 0.5}
# create plot
fig = plot_compare_evokeds(evokeds,
                           linestyles={'A Cue': '-', 'B Cue': '--'},
                           styles={'A Cue': {"linewidth": 2.},
                                   'B Cue': {"linewidth": 2.}},
                           show_legend='upper center',
                           ylim=dict(eeg=[0, 4]),
                           cmap=cmap,
                           colors=colors)
fig.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig.axes[0].set_yticks(list(np.arange(0, 5, 1)), minor=False)
fig.axes[0].add_collection(collection)
fig.axes[0].annotate('t1', xy=(.074, 4.), weight="bold")
fig.axes[0].annotate('t2', xy=(.16, 4.), weight="bold")
fig.axes[0].annotate('t3', xy=(.27, 4.), weight="bold")
fig.axes[0].annotate('t4', xy=(.44, 4.), weight="bold")
fig.axes[0].annotate('t5', xy=(2.175, 4.), weight="bold")
fig.axes[0].spines['top'].set_visible(False)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['left'].set_bounds(0, 4)
fig.axes[0].spines['bottom'].set_bounds(-.25, 2.5)
fig.axes[0].xaxis.set_label_coords(.5, -.13)
fig.set_size_inches(9, 4)
fig.subplots_adjust(bottom=0.15)
fig.savefig(op.join(output_path, 'GFP_evokeds.pdf'), dpi=300)

# --- 3.3) plot cue ERPs ---------
# electrodes to plot
electrodes =  ['FCz', 'Pz', 'PO8']

for electrode in electrodes:
    # electrode in question
    pick = Grand_Average_A.ch_names.index(electrode)
    cmap = 'magma'
    colors = {'A Cue': 0, 'B Cue': 0.6}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax = plot_compare_evokeds({'A Cue': Grand_Average_A.copy().crop(tmin=-.5),
                               'B Cue': Grand_Average_B.copy().crop(tmin=-.5)},
                              ylim=dict(eeg=[-8, 6]),
                              picks=pick,
                              invert_y=False,
                              cmap=cmap,
                              colors=colors,
                              show_legend=3,
                              show_sensors=2,
                              truncate_xaxis=False,
                              axes=ax)

    ax.axes[0].fill_between(times,
                            upper_a[pick]*1e6,
                            lower_a[pick]*1e6,
                            alpha=0.25,
                            color=cm.magma(0))

    ax.axes[0].fill_between(times,
                            upper_b[pick]*1e6,
                            lower_b[pick]*1e6,
                            alpha=0.25,
                            color=cm.magma(0.6))
    #
    # ax = plot_compare_evokeds({'A Cue': [val.crop(tmin=-.5) for val in
    #                                      erps_a_cue.values()],
    #                            'B Cue': [val.crop(tmin=-.5) for val in
    #                                      erps_b_cue.values()]},
    #                           ylim=dict(eeg=[-6, 6]),
    #                           picks=pick, invert_y=True,
    #                           cmap=cmap,
    #                           colors=colors,
    #                           show_legend=2,
    #                           show_sensors=3,
    #                           truncate_xaxis=False,
    #                           axes=ax,
    #                           ci=.95)

    ax.axes[0].set_xlim(-.5, 2.5)
    ax.axes[0].xaxis.set_ticks(np.arange(-.5, 2.5+0.1, .25))
    plt.plot()
