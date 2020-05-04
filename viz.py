# -*- coding: utf-8 -*-
"""Utility functions for plotting.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import stats
from sklearn.preprocessing import normalize


def plot_z_scores(z_scores, channels, bads=None, cmap='inferno', show=False):

    cmap = cm.get_cmap(cmap)

    # plot results
    z_colors = normalize(
        np.abs(z_scores).reshape((1, z_scores.shape[0]))).ravel()

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    fig, ax = plt.subplots(figsize=(20, 6))
    if z_scores.max() < 5.0:
        y_lim = 5
    else:
        y_lim = int(z_scores.max() + 2)

    for i in range(z_scores.shape[0]):
        ch = channels[i]
        # show channel names in red if bad by correlation
        if ch in bads:
            col = 'crimson'
        else:
            col = 'k'
        ax.axhline(y=5.0, xmin=-1.0, xmax=65,
                   color='crimson', linestyle='dashed', linewidth=2.0)
        ax.text(-5.0, 5.0, 'crit. Z-score', fontsize=14,
                verticalalignment='center', horizontalalignment='center',
                color='crimson', bbox=props)
        ax.bar(i, np.abs(z_scores[i]), width=0.9, color=cmap(z_colors[i]))
        ax.text(i, np.abs(z_scores[i]) + 0.25, ch, color=col,
                fontweight='bold', fontsize=9,
                ha='center', va='center', rotation=45)
    ax.set_ylim(0, y_lim)
    ax.set_xlim(-1, 64)

    plt.title('EEG channel deviation', {'fontsize': 15, 'fontweight': 'bold'})
    plt.xlabel('Channels', {'fontsize': 13}, labelpad=10)
    plt.ylabel('Abs. Z-Score', {'fontsize': 13}, labelpad=10)

    plt.xticks([])
    plt.yticks(fontsize=12)

    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(0, y_lim)

    plt.close(fig)

    return fig.show() if show else fig


def within_subject_cis(insts, n_cond = 2):
    # see Morey (2008): Confidence Intervals from Normalized Data:
    # A correction to Cousineau (2005)

    if len(np.unique([len(i) for i in insts])) > 1:
        raise ValueError('inst must be of same length')

    # correction factor for number of conditions
    n_cond = len(insts)
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
