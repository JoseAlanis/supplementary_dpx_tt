# --- jose c. garcia alanis
# --- utf-8
# --- Python 3.6.2
#
# --- eeg prepossessing - dpx tt
# --- version jan 2018
#
# --- visualise erp results
# --- export figures to .pdf

# ==================================================================================================
# ------------------------------ Import relevant extensions ----------------------------------------
import glob
import os

import numpy as np

import mne
# from mne.time_frequency import tfr_morlet

import pickle

import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.rcParams['lines.linewidth'] = 1.0
sns.set_style('ticks', {"xtick.major.size": 5, "ytick.major.size": 5})
sns.set_context('paper', font_scale=1.3, rc={'lines.linewidth': 1})

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH to data
data_path = '/Volumes/TOSHIBA/manuscripts_and_data/dpx_tt/eeg/'


# ========================================================================
# execute this cell to load previously saved epochs data
with open(data_path + 'dpx_mne_results/tfr_scores/erp_data_dict.pkl', 'rb') as erp:
    erp_data_dict = pickle.load(erp)

# extract epochs data
cue_a_epochs = erp_data_dict['allEpochs_A'][0]
cue_b_epochs = erp_data_dict['allEpochs_B'][0]

# clean up
del erp_data_dict, erp

# .copy().apply_baseline(baseline=(-0.3, -0.05))
# .copy().apply_baseline(baseline=(-0.3, -0.05))

# --- compute individual erps (cue A) ---
individual_erps_a = [cue_a_epochs[i].average() for i in range(0, len(cue_a_epochs))]
# grand average ERP cue A
ERP_A = mne.grand_average(individual_erps_a)

# --- compute individual erps (cue B) ---
individual_erps_b = [cue_b_epochs[i].average() for i in range(0, len(cue_b_epochs))]
# grand average ERP cue B
ERP_B = mne.grand_average(individual_erps_b)


# === PLOT ERPs ===
# arguments for time series plot
ts_args = dict(gfp=False,
               time_unit='s',
               ylim=dict(eeg=[-9, 9]),
               xlim=[-.25, 2.5])
# arguments fot the topographical maps
topomap_args = dict(sensors=False,
                    time_unit='ms',
                    vmin=-7, vmax=7,
                    average=0.05)
# times to plot
ttp = [.100, .200, .300, .500, .700, 2.45]

# save figure cue A
fig_a = ERP_A.plot_joint(times=ttp,
                         ts_args=ts_args,
                         topomap_args=topomap_args,
                         title='Average Evoked Activity - Cue A')
fig_a.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig_a.axes[0].set_yticks(list(np.arange(-9, 9.5, 4.5)), minor=False)
fig_a.axes[0].axhline(y=0, xmin=-.5, xmax=2.5, color='black', linestyle='dotted')
fig_a.axes[0].axvline(x=0, ymin=-5, ymax=5, color='black', linestyle='dotted')
# fig_a.axes[0].annotate('GFP', xy=(-.25, -8))
sns.despine(offset=5, trim=True)
fig_a.set_size_inches(9, 4.5)
fig_a.savefig(data_path + 'dpx_mne_results/figures/ERP_A.pdf', dpi=300)

# save figure cue B
fig_b = ERP_B.plot_joint(times=ttp,
                         ts_args=ts_args,
                         topomap_args=topomap_args,
                         title='Average Evoked Activity - Cue B')
fig_b.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig_b.axes[0].set_yticks(list(np.arange(-9, 9.5, 4.5)), minor=False)
fig_b.axes[0].axhline(y=0, xmin=-.5, xmax=2.5, color='black', linestyle='dotted')
fig_b.axes[0].axvline(x=0, ymin=-5, ymax=5, color='black', linestyle='dotted')
# fig_b.axes[0].annotate('GFP', xy=(-.25, -8))
sns.despine(offset=5, trim=True)
fig_b.set_size_inches(9, 4.5)
fig_b.savefig(data_path + 'dpx_mne_results/figures/ERP_B.pdf', dpi=300)


# === PLOT plot difference between brain waves ===
# compute difference
diff_wave = mne.combine_evoked([ERP_B, ERP_A], weights=[1, -1])
# plot args
kwargs = dict(vmin=-5, vmax=5,
              average=0.1,
              sensors=False,
              time_unit='ms')
# create plot first chunk
fig, ax = plt.subplots(1, 7, figsize=(12, 2.5))

diff_wave.plot_topomap(axes=ax[0], times=[0.2], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[1], times=[0.3], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[2], times=[0.4], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[3], times=[0.5], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[4], times=[0.6], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[5], times=[0.7], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[6], times=[0.8], colorbar=False, **kwargs)
for ax, title in zip(ax[:7], ['0.1 s', '0.2 s', '0.3 s', '0.4 s', '0.5 s', '0.6 s', '0.7 s', '0.8 s']):
    ax.set_title(title)
plt.show()
fig.set_size_inches(10, 2.5)
fig.savefig(data_path + 'dpx_mne_results/figures/Diff_Topo_1.pdf', dpi=300)

# create plot second chunk
fig, ax = plt.subplots(1, 7, figsize=(12, 2.5))

diff_wave.plot_topomap(axes=ax[0], times=[1.2], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[1], times=[1.4], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[2], times=[1.6], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[3], times=[1.8], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[4], times=[2], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[5], times=[2.2], colorbar=False, **kwargs)
diff_wave.plot_topomap(axes=ax[6], times=[2.4], colorbar=False, **kwargs)
for ax, title in zip(ax[:7], ['1.2 s', '1.4 s', '1.6 s', '1.8 s', '2.0 s', '2.2 s', '2.2 s', '2.4 s']):
    ax.set_title(title)
plt.show()
fig.set_size_inches(10, 2.5)
fig.savefig(data_path + 'dpx_mne_results/figures/Diff_Topo_2.pdf', dpi=300)


# === PLOT GFP ===
gfp_times = {'t1': [[0.07, 0], 0.07],
             't2': [[0.14, 0], 0.10],
             't3': [[0.24, 0], 0.12],
             't4': [[0.36, 0], 0.24],
             't5': [[2, 0], 0.4]}
# create annotation patches
patches = []
for key, val in gfp_times.items():
    rect = mpatches.Rectangle(val[0], val[1], 4, ec="none")
    patches.append(rect)
# use viridis colors
colors = np.linspace(0, 1, len(patches))
collection = PatchCollection(patches, cmap=plt.cm.viridis, alpha=0.25)
collection.set_array(np.array(colors))
# plot GFP and save figure
evokeds = {'Cue A': ERP_A.copy().crop(tmin=-.25), 'Cue B': ERP_B.copy().crop(tmin=-.25)}
fig_ab = mne.viz.plot_compare_evokeds(evokeds,
                                      linestyles={'Cue A': '-', 'Cue B': '--'},
                                      styles={'Cue A': {"linewidth": 1.5},
                                              'Cue B': {"linewidth": 1.5}},
                                      show_legend='upper center',
                                      ylim=dict(eeg=[0, 4]),
                                      cmap='gray',
                                      colors={'Cue A': 0, 'Cue B': 0.1})
fig_ab.axes[0].set_xticks(list(np.arange(-.25, 2.55, .25)), minor=False)
fig_ab.axes[0].set_yticks(list(np.arange(0, 5, 1)), minor=False)
fig_ab.axes[0].add_collection(collection)
fig_ab.axes[0].annotate('t1', xy=(.074, 4.), weight="bold")
fig_ab.axes[0].annotate('t2', xy=(.16, 4.), weight="bold")
fig_ab.axes[0].annotate('t3', xy=(.27, 4.), weight="bold")
fig_ab.axes[0].annotate('t4', xy=(.44, 4.), weight="bold")
fig_ab.axes[0].annotate('t5', xy=(2.175, 4.), weight="bold")
sns.despine(offset=5, trim=True)
fig_ab.axes[0].xaxis.set_label_coords(.5, -.2)
fig_ab.set_size_inches(9, 3.5)
fig_ab.subplots_adjust(bottom=0.175)
fig_ab.savefig(data_path + 'dpx_mne_results/figures/GPF_AB.pdf', dpi=300)


# === PLOT TF INDUCED RESULTS ===============================

Induced_A_BC = [Induced_A[i].copy().apply_baseline(mode='logratio',
                                                   baseline=(-1., -.5))
                for i in range(0, len(Induced_A))]

TF_Induced_A_BC = [Induced_A_BC[i].data*10 for i in range(0, len(Induced_A_BC))]

Induced_A_BC = [mne.time_frequency.AverageTFR(data=TF_Induced_A_BC[i],
                                              times=Induced_A_BC[i].times,
                                              freqs=Induced_A_BC[i].freqs,
                                              info=Induced_A_BC[i].info,
                                              nave=Induced_A_BC[i].nave)
                for i in range(0, len(Induced_A_BC))]

TFR_Induced_A = mne.grand_average(Induced_A_BC)



topomap_args = dict(sensors=False, vmin=-4.5, vmax=4.5)
TFR_Induced_A.plot_joint(baseline=None,
                         vmin=-4.5, vmax=4.5,
                         tmin=-.3, tmax=2.4,
                         timefreqs={(.2, 5): (0.2, 2),
                                    (.4, 20): (0.2, 10),
                                    (.4, 11): (0.2, 4),
                                    (.9, 16): (0.2, 2),
                                    (2.3, 16): (0.2, 2)},
                         topomap_args=topomap_args,
                         colorbar=True,
                         yscale='log')

TFR_Induced_A.plot_topo(baseline=None,
                        title='Average power', tmin=-.5, tmax=2.5)

TFR_Induced_A.plot([61], baseline=None,
                   tmin=-.5, tmax=2.3,
                   title=TFR_Induced_A.ch_names[61])








# === PLOT TF ERP RESULTS ===============================
# Total A


topomap_args = dict(sensors=False, vmin=-4, vmax=4)
TFR_Total_A.plot_joint(baseline=None,
                       vmin=-4, vmax=4,
                       tmin=-.3, tmax=2.5,
                       timefreqs={(.2, 5): (0.2, 2),
                                  (.4, 20): (0.2, 10),
                                  (.4, 11): (0.2, 4),
                                  # (.9, 16): (0.2, 2),
                                  (2.3, 16): (0.2, 2)},
                       topomap_args=topomap_args,
                       colorbar=True,
                       yscale='log')

TFR_Total_A_fig = TFR_Total_A.plot_joint(baseline=None,
                                         vmin=-5, vmax=5,
                                         tmin=-.5, tmax=2.4,
                                         timefreqs={(.2, 5): (0.2, 2),
                                                    (.4, 20): (0.2, 10),
                                                    (.4, 11): (0.2, 4),
                                                    (.9, 16): (0.2, 2),
                                                    (2.3, 16): (0.2, 2)},
                                         topomap_args=topomap_args,
                                         colorbar=True,
                                         yscale='log',
                                         show=False)

TFR_Total_A_fig.set_size_inches(12, 6)
TFR_Total_A_fig.savefig(data_path + 'figures/TFR_Total_A.pdf', dpi=300)


# Total B
Total_B_BC = [Total_B[i].copy().apply_baseline(mode='logratio',
                                               baseline=(-1., -.5))
              for i in range(0, len(Total_B))]

TF_Total_B_BC = [Total_B_BC[i].data*10 for i in range(0, len(Total_B_BC))]

Total_B_BC = [mne.time_frequency.AverageTFR(data=TF_Total_B_BC[i],
                                            times=Total_B_BC[i].times,
                                            freqs=Total_B_BC[i].freqs,
                                            info=Total_B_BC[i].info,
                                            nave=Total_B_BC[i].nave)
              for i in range(0, len(Total_B_BC))]

TFR_Total_B = mne.grand_average(Total_B_BC)

TFR_Total_B.plot_topo(baseline=None,
                      title='Average power', tmin=-.5, tmax=2.3)

TFR_Total_B.plot([11], baseline=None,
                 tmin=-.5, tmax=2.3,
                 title=TFR_Total_B.ch_names[11])

topomap_args = dict(sensors=False, vmin=-5, vmax=5)
TFR_Total_B.plot_joint(baseline=None,
                       vmin=-5, vmax=5,
                       tmin=-.5, tmax=2.4,
                       timefreqs={(.2, 5): (0.2, 2),
                                  (.5, 20): (0.2, 10),
                                  (.5, 11): (0.2, 4),
                                  (1, 16): (0.2, 2),
                                  (2.3, 16): (0.2, 2)},
                       topomap_args=topomap_args,
                       colorbar=True,
                       yscale='log')

TFR_Total_B_fig = TFR_Total_B.plot_joint(baseline=None,
                                         vmin=-5, vmax=5,
                                         tmin=-.5, tmax=2.4,
                                         timefreqs={(.2, 5): (0.2, 2),
                                                    (.5, 20): (0.2, 10),
                                                    (.5, 11): (0.2, 4),
                                                    (1, 16): (0.2, 2),
                                                    (2.3, 16): (0.2, 2)},
                                         topomap_args=topomap_args,
                                         colorbar=True,
                                         yscale='log',
                                         show=False)

TFR_Total_B_fig.set_size_inches(12, 6)
TFR_Total_B_fig.savefig(data_path + 'figures/TFR_Total_B.pdf', dpi=300)




# === PLOT TF ITC RESULTS ===============================
# ITC A
TFR_ITC_A = mne.grand_average(ITC_A)

TFR_ITC_A.plot_topo(baseline=None,
                    title='Average power', tmin=-.5, tmax=2.3)

topomap_args = dict(sensors=False, vmin=0, vmax=.8)
TFR_ITC_A.plot_joint(baseline=None,
                     vmin=0, vmax=.8,
                     tmin=-.5, tmax=2.4,
                     timefreqs={(.2, 5): (0.2, 2)},
                     topomap_args=topomap_args,
                     colorbar=True,
                     yscale='log',
                     cmap='Reds')

TFR_ITC_A_fig = TFR_ITC_A.plot_joint(baseline=None,
                                     vmin=0, vmax=.8,
                                     tmin=-.5, tmax=2.4,
                                     timefreqs={(.2, 5): (0.2, 2)},
                                     topomap_args=topomap_args,
                                     colorbar=True,
                                     yscale='log',
                                     cmap='Reds',
                                     show=False)

TFR_ITC_A_fig.set_size_inches(12, 6)
TFR_ITC_A_fig.savefig(data_path + 'figures/TFR_ITC_A.pdf', dpi=300)

# ITC B
TFR_ITC_B = mne.grand_average(ITC_B)

TFR_ITC_B.plot_topo(baseline=None,
                    title='Average power', tmin=-.5, tmax=2.3)

topomap_args = dict(sensors=False, vmin=0, vmax=.7)
TFR_ITC_B.plot_joint(baseline=None,
                     vmin=0, vmax=.7,
                     tmin=-.5, tmax=2.4,
                     timefreqs={(.2, 5): (0.2, 2),
                                (.7, 3): (0.2, 1)},
                     topomap_args=topomap_args,
                     colorbar=True,
                     yscale='log', cmap='Reds')

TFR_ITC_B_fig = TFR_ITC_B.plot_joint(baseline=None,
                                     vmin=0, vmax=.8,
                                     tmin=-.5, tmax=2.4,
                                     timefreqs={(.2, 5): (0.2, 2)},
                                     topomap_args=topomap_args,
                                     colorbar=True,
                                     yscale='log',
                                     cmap='Reds',
                                     show=False)

TFR_ITC_B_fig.set_size_inches(12, 6)
TFR_ITC_B_fig.savefig(data_path + 'figures/TFR_ITC_B.pdf', dpi=300)
