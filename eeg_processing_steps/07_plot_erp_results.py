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
erps_a_cue = {subj: cues_dict[subj]['Correct A'].apply_baseline((-.3, -.05)).average()
              for subj in cues_dict}
# cue B
erps_b_cue = {subj: cues_dict[subj]['Correct B'].apply_baseline((-.3, -.05)).average()
              for subj in cues_dict}

subject_erp = {subj: cues_dict[subj]['Correct B', 'Correct A'].apply_baseline((-.3, -.05)).average()
               for subj in cues_dict}

# save times
times = cues_dict['001']['Correct A'].times

# ---
import mne  # noqa
test_epochs = cues_dict['001']['Correct A']

noise_cov = mne.compute_covariance(
    test_epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, test_epochs.info)

evoked = test_epochs.average().pick('eeg')
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(0.10, 0.20, 5), ch_type='eeg',
                    time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')

del test_epochs  # to save memory

# Read the forward solution and compute the inverse operator
fname_fwd = '/Volumes/TOSHIBA/manuscripts_and_data/dpx_tt/derivatives/coreg/sub-001/sub-001_fsol-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd)

# make an MEG inverse operator
info = evoked.info
inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov,
                                                          loose=0.2, depth=0.8)
del fwd

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
                                               method=method, pick_ori=None,
                                               return_residual=True, verbose=True)

plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

vertno_max, time_max = stc.get_peak(hemi='rh')

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='med',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

morph = mne.compute_source_morph(
    src=inverse_operator['src'], subject_from=stc.subject,
    subject_to='fsaverage', spacing=5,  # to ico-5
    subjects_dir=subjects_dir)
# morph data
stc_fsaverage = morph.apply(stc)

brain = stc_fsaverage.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)
del stc_fsaverage

stc_vec = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2,
                                         method=method, pick_ori='vector')
brain = stc_vec.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Vector solution', 'title', font_size=20)
del stc_vec

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
ttp = [.100, .180, .310, .500, .680, 1.00, 2.45]
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='s',
                    vmin=8, vmax=-8,
                    average=0.05,
                    outlines='skirt')

# --- 3.1) plot evoked activity ---------
# A cues
fig = Grand_Average_A.plot_joint(times=ttp,
                                 ts_args=ts_args,
                                 topomap_args=topomap_args,
                                 title='Average Evoked Activity - Cue A')
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
fig.axes[0].xaxis.set_label_coords(.5, -.175)
fig.set_size_inches(8, 4)
fig.savefig(op.join(output_path, 'Evoked_A_Cue.pdf'), dpi=300)

# B cues
fig = Grand_Average_B.plot_joint(times=ttp,
                                 ts_args=ts_args,
                                 topomap_args=topomap_args,
                                 title='Average Evoked Activity - Cue B')
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
fig.axes[0].xaxis.set_label_coords(.5, -.175)
fig.set_size_inches(8, 4)
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
fig, ax = plt.subplots(figsize=(8, 4))
plot_compare_evokeds(evokeds,
                     axes=ax,
                     linestyles={'A Cue': '-', 'B Cue': '--'},
                     styles={'A Cue': {"linewidth": 2.},
                             'B Cue': {"linewidth": 2.}},
                     legend='upper center',
                     ylim=dict(eeg=[0, 4]),
                     cmap=cmap,
                     colors=colors,
                     )
ax.set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
ax.set_yticks(list(np.arange(0, 5, 1)), minor=False)
ax.add_collection(collection)
ax.annotate('t1', xy=(.074, 4.), weight="bold")
ax.annotate('t2', xy=(.16, 4.), weight="bold")
ax.annotate('t3', xy=(.27, 4.), weight="bold")
ax.annotate('t4', xy=(.44, 4.), weight="bold")
ax.annotate('t5', xy=(2.175, 4.), weight="bold")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(0, 4)
ax.spines['bottom'].set_bounds(-.25, 2.5)
ax.xaxis.set_label_coords(.5, -.13)
fig.subplots_adjust(bottom=0.15)
fig.savefig(op.join(output_path, 'GFP_evokeds.pdf'), dpi=300)

# --- 3.3) plot cue ERPs ---------
# electrodes to plot
electrodes = ['PO8', 'PO7']

for electrode in electrodes:
    # electrode in question
    pick = Grand_Average_A.ch_names.index(electrode)
    cmap = 'magma'
    colors = {'A Cue': 0, 'B Cue': 0.6}
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_compare_evokeds({'A Cue': Grand_Average_A.copy().crop(tmin=-.5),
                          'B Cue': Grand_Average_B.copy().crop(tmin=-.5)},
                         ylim=dict(eeg=[-8, 6]),
                         picks=pick,
                         invert_y=False,
                         cmap=cmap,
                         colors=colors,
                         legend='lower left',
                         show_sensors=2,
                         truncate_xaxis=False,
                         axes=ax)

    ax.fill_between(times,
                    upper_a[pick]*1e6,
                    lower_a[pick]*1e6,
                    alpha=0.2,
                    color=cm.magma(0))

    ax.fill_between(times,
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

    ax.set_xlim(-.5, 2.5)
    ax.xaxis.set_ticks(np.arange(-.5, 2.5+0.1, .25))
    plt.plot()
    fig.savefig(op.join(output_path, 'Cue_ERP_%s.pdf') % electrode, dpi=300)