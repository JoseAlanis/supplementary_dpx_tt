# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7 / mne 0.19.2
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- import data,
# --- convert to bids format

# ========================================================================
# ------------------- import relevant extensions -------------------------
import glob
import os.path as op
from os import mkdir

import re

from mne.datasets import fetch_fsaverage
from mne.channels import make_standard_montage
from mne.io import read_raw_bdf
from mne.viz import plot_alignment, plot_montage
from mne import make_forward_solution, sensitivity_map, \
    write_forward_solution

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
# --- 1) set eeg channel names and locations ---------------
# get standard 10-20 channel information
montage = make_standard_montage(kind='standard_1020')

# visualize channel arrangement
plot_montage(montage)

# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8',
           'EOGV_oben', 'EOGV_unten', 'EOGH_rechts', 'EOGH_links']

# --- 2) get sample file for co-registration  --------------
# file path and name
file = files[0]
filepath, filename = op.split(file)

# subject in question
subj = re.findall(r'\d+', filename)[0].rjust(3, '0')
subject = 'sub-%s' % subj

# --- 3) import the data -----------------------------------
raw = read_raw_bdf(file,
                   preload=True,
                   exclude=exclude)
# apply montage to data
raw.set_montage(montage)
# set eeg reference
raw.set_eeg_reference(projection=True)

# --- 4) create directory for saving -----------------------
if not op.exists(op.join(output_path, subject)):
    mkdir(op.join(output_path, subject))
# save file
raw.save(op.join(output_path, subject, subject + '_for_coreg-raw.fif'),
         overwrite=True)

# --- 5) load fsaverage files ------------------------------
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# get the transformation file for montage
trans = op.join(output_path, subject, subject + '-std1020-trans.fif')

# sources and boundary element model computed from fsaverage
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# --- 6) check co-registration -----------------------------
# plot alignment with head surface
plot_alignment(raw.info, src=src, eeg=['original', 'projected'],
               trans=trans, dig=True, mri_fiducials=True,
               coord_frame='mri')

# --- 7) compute forward solution --------------------------
fwd = make_forward_solution(raw.info, trans=trans, src=src,
                            bem=bem, eeg=True, mindist=5.0, n_jobs=2)
print(fwd)

# --- 8) check the eeg sensitivity ma ----------------------
eeg_map = sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             hemi='both',
             clim=dict(lims=[5, 50, 100]), surface='inflated')

# --- 9) save forward solution
file_name = op.join(output_path, subject, subject + '_fsol-1020-fwd.fif')
write_forward_solution(file_name, fwd=fwd, overwrite=True)
