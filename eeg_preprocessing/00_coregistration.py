
# ========================================================================
# ------------------- import relevant extensions -------------------------
import glob
import os.path as op
from os import mkdir

import re

from mne.datasets import fetch_fsaverage
from mne.channels import read_montage
from mne.io import read_raw_bdf
from mne.viz import plot_alignment, plot_montage
from mne import make_forward_solution, sensitivity_map

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if op.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# path to eeg files
data_path = op.join(root_path, 'sub-*')

# path for saving output
derivatives_path = op.join(root_path, 'derivatives')

# create directory for derivatives
if not op.isdir(derivatives_path):
    mkdir(derivatives_path)
    mkdir(op.join(derivatives_path, 'coreg'))

# path for saving script output
output_path = op.join(derivatives_path, 'coreg')

# files to be analysed
files = sorted(glob.glob(op.join(data_path, 'eeg/*.bdf')))

# ========================================================================
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
# trans = op.join(fs_dir, 'bem', 'fsaverage-trans.fif')
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

trans = op.join(output_path, 'sub-001/sub-001-trans.fif')


# eeg channel names and locations
montage = read_montage(kind='standard_1020')

# plot_montage(montage, kind='3d')

# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8',
           'EOGV_oben', 'EOGV_unten', 'EOGH_rechts', 'EOGH_links']

# --- 1) set up paths and file names -----------------------
file = files[0]
filepath, filename = op.split(file)

# subject in question
subj = re.findall(r'\d+', filename)[0].rjust(3, '0')

# --- 2) import the data -----------------------------------
raw = read_raw_bdf(file,
                   preload=True,
                   exclude=exclude)

raw.set_montage(montage)

# create directory for save
if not op.exists(op.join(output_path, 'sub-%s' % subj)):
    mkdir(op.join(output_path, 'sub-%s' % subj))

raw.save(op.join(output_path, 'sub-' + str(subj),
                 'sub-%s_for_coreg-raw.fif' % subj),
         overwrite=True)

plot_alignment(raw.info, src=src, eeg=['original', 'projected'],
               trans=trans, dig=True, mri_fiducials=True)

fwd = make_forward_solution(raw.info, trans=trans, src=src,
                            bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)

# for illustration purposes use fwd to compute the sensitivity map
eeg_map = sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             hemi='both',
             clim=dict(lims=[5, 50, 100]), surface='inflated')



