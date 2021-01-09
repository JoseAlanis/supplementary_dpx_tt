"""
=============================================
Plot results of multivariate pattern analysis
=============================================

Creates figures to show classifier performance at multiple time points of
the of the EEG epoch.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import seaborn as sns

from config import fname, subjects
from mvpa_stats import get_stats_lines, get_dfs, plot_image

from mne import read_epochs

# exclude subjects 51
subjects = subjects[subjects != 51]

##############################################################################
# 1) import a generic file to use as template
input_file = fname.output(subject=subjects[0],
                          processing_step='cue_epochs',
                          file_type='epo.fif')
cue_epo = read_epochs(input_file, preload=True)
cue_epo = cue_epo.crop(tmin=-0.5, include_tmax=False)

##############################################################################
# 2) import mvpa results

# load GAT scores
scores = np.load(fname.results + '/gat_scores_ridge.npy')

# load p values
p_vals = np.load(fname.results + '/p_vals_gat_ridge.npy')

##############################################################################
# 3) create generalisation across time (GAT) matrix figure
data = scores.copy()
fig, axes = plt.subplots(figsize=(6, 4.5))
plot_image(data.mean(0),
           cue_epo.times,
           mask=p_vals < 0.01,
           ax=axes, vmax=.7, vmin=.3,
           draw_mask=True, draw_contour=True, colorbar=True,
           draw_diag=True, draw_zerolines=True, xlabel="Time (s)",
           ylabel="Time (s)",
           cbar_unit="%", cmap="RdBu_r", mask_cmap="RdBu_r", mask_alpha=.95)
axes.spines['top'].set_bounds(-0.5, 2.5)
axes.spines['right'].set_bounds(-0.5, 2.5)
axes.spines['left'].set_bounds(-0.5, 2.5)
axes.spines['bottom'].set_bounds(-0.5, 2.5)
axes.set_xticks(list(np.arange(-.5, 2.55, .5)), minor=False)
axes.set_yticks(list(np.arange(-.5, 2.55, .5)), minor=False)
fig.savefig(fname.figures + '/gat_matrix.pdf', dpi=300)


##############################################################################
# 4) Plot classifier performance for specific time slices of interest
test_times = dict(N170=[0.17, 0.27], LPC=[0.50, 0.60], CNV=[0.95, 1.05])

# compute significance for those time slices
stats_dict = get_stats_lines(scores, times=cue_epo.times, test_times=test_times)

colors = np.linspace(0.2, 0.8, len(test_times.values()))
cmap = cm.get_cmap('inferno')

# create figure
for df_type in ['diag', False]:

    if df_type == 'diag':
        title = 'Diagonal decoding performance'
        name = 'diagonal_performance'
    else:
        title = 'Component generalization across time'
        name = 'component_performance'
    lw_b = 0.45
    up_b = 0.70

    palette = [cmap(colors[i]) for i, val in enumerate(test_times.values())]
    fig, axes = plt.subplots(figsize=(9, 4.5))
    onsets = {k: v[0] for (k, v) in test_times.items()}
    axes.bar(onsets.values(), 1, width=0.1, alpha=0.15, align='edge', color=palette)
    # for i, val in enumerate({k: v[0] for (k, v) in test_times.items()}):
    #     axes.bar(val[0], 1, width=val[1], alpha=0.15,
    #              align='edge', color=cmap(colors[i]))

    if df_type == 'diag':
        sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
                     color='k',
                     y='Accuracy (%)',
                     x='Time (s)',
                     ci=95,
                     ax=axes)

        for t in stats_dict['diag'][1]:
            axes.scatter(t, 0.45, marker='_', color='k', s=1.0)
        for t in stats_dict['diag'][2]:
            axes.scatter(t, 0.45, marker='|', color='k', s=25.0)

    else:
        sns.lineplot(data=get_dfs(stats_dict, df_type=df_type),
                     hue='Component',
                     y='Accuracy (%)',
                     x='Time (s)',
                     ci=95,
                     palette=palette,
                     ax=axes)

        components = stats_dict.keys()
        components = [c for c in components if c != 'diag']
        max_off = (len(onsets) * 0.5) / 100
        offsets = np.linspace(0.45, 0.45+max_off, len(onsets)) - np.linspace(
            0.45, 0.45+max_off, len(onsets)).mean()

        for n_comp, comp in enumerate(components):
            for t in stats_dict[comp][1]:
                axes.scatter(t, 0.45+offsets[n_comp], marker='_', color=palette[n_comp], s=1.0)
            for t in stats_dict[comp][2]:
                axes.scatter(t, 0.45+offsets[n_comp], marker='|', color=palette[n_comp], s=25.0)

    axes.set_title(title, pad=10.0)
    axes.set_xlabel('Time (s)', labelpad=10.0)
    axes.set_ylabel('Accuracy (%)', labelpad=10.0)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_bounds(lw_b, up_b)
    axes.spines['bottom'].set_bounds(-0.5, 2.5)
    axes.set_ylim(lw_b - 0.015, up_b + 0.025)

    axes.axhline(y=0.5, xmin=-.5, xmax=2.5,
                 color='black', linestyle='dashed', linewidth=.8)
    axes.axvline(x=0.0, ymin=0, ymax=1.0,
                 color='black', linestyle='dashed', linewidth=.8)

    fig.savefig(fname.figures + '/%s_gat_ridge.pdf' % name, dpi=300)
