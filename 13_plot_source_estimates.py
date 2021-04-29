"""
=============================
Plot LCMV beamforming results
=============================

Plot results of source estimation algorithm.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os.path as op

import pickle

import numpy as np

from nilearn.plotting import plot_glass_brain

import matplotlib.pyplot as plt
from matplotlib import colors

from mne.datasets import fetch_fsaverage
from mne.viz import plot_brain_colorbar
from mne import read_source_spaces

# All parameters are defined in config.py
from config import fname

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
fname_t1_fsaverage = op.join(subjects_dir, 'fsaverage', 'mri',
                             'brain.mgz')
fname_src_fsaverage = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'
src_fs = read_source_spaces(fname_src_fsaverage)

###############################################################################
# 1) load previously computed source estimates
with open(fname.results + '/stcs_lcmv.pkl', 'rb') as f:
    stcs = pickle.load(f)

###############################################################################
# 2) compute average of source estimates

# template
stc_average_a = stcs['cue_a'][0]
stc_average_b = stcs['cue_b'][0]

# fill template with averaged data
stc_average_a.data = np.average([s.copy().data for s in stcs['cue_a']], axis=0)
stc_average_b.data = np.average([s.copy().data for s in stcs['cue_b']], axis=0)

# # free space in memory
# del x

###############################################################################
# 2) compute average of source estimates
lims = [0.05, 0.075, 0.1]
clim = dict(kind='value', lims=lims)
scale_pts = np.array(lims)
colormap = plt.get_cmap('magma_r')
colormap = colormap(
    np.interp(np.linspace(-1, 1, 256),
              scale_pts / scale_pts[2],
              [0, 0.5, 1]))
colormap = colors.ListedColormap(colormap)

peak_stc_b = stc_average_b.copy().crop(tmin=0.2, tmax=0.2)
peak_stc_a = stc_average_b.copy().crop(tmin=0.2, tmax=0.2)
peak_stc = peak_stc_b - peak_stc_b

img = peak_stc.as_volume(src_fs, mri_resolution=False)

fig = plt.figure(figsize=(6, 2.5))
axes = [plt.subplot2grid((4, 16), (0, 0), rowspan=4, colspan=14),
        plt.subplot2grid((4, 16), (1, 15), rowspan=2, colspan=1)]

plot_glass_brain(img,
                 title='t = 0.50s, Cue = A',
                 draw_cross=False,
                 annotate=True,
                 colorbar=False,
                 cmap=colormap,
                 threshold=lims[0],
                 vmax=lims[-1],
                 axes=axes[0])
plot_brain_colorbar(axes[1], clim, 'magma_r', label='Activation (NAI)',
                    bgcolor='white')
fig.subplots_adjust(
    left=0.05, right=0.9, bottom=0.01, top=0.9, wspace=0.5, hspace=0.1)
fig.savefig(fname.figures + '/lcmv_A_500ms_lower.pdf', dpi=300)
