"""
==================================
Exploratory analysis of cue epochs
==================================

Compute descriptive statistics and exploratory analysis plots
for cue locked epochs.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import matplotlib.pyplot as plt

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
electrode = 'FCz'
pick = ga_a_cue.ch_names.index(electrode)

fig, ax = plt.subplots(figsize=(8, 4))

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
ax.set_xlim(-0.500, 2.500)
plot_compare_evokeds({'A': ga_a_cue.copy().crop(-0.5, 2.5), 'B': ga_b_cue.copy().crop(-0.5, 2.5)},
                     picks=pick,
                     invert_y=True,
                     ylim=dict(eeg=[-10, 5]),
                     colors={'A': 'k', 'B': 'crimson'},
                     axes=ax)

ga_b_cue.plot_image(xlim=[-0.5, 2.5], clim=dict(eeg=[-8, 8]))
ga_a_cue.plot_image(xlim=[-0.5, 2.5], clim=dict(eeg=[-8, 8]))

