# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- ica decomposition, find eog artifacts,
# --- export ica solution

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall

from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import ICA

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
data_path = op.join(derivatives_path, 'artifact_detection')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'ica')):
    mkdir(op.join(derivatives_path, 'ica'))

# path for saving output
output_path = op.join(derivatives_path, 'ica')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# ========================================================================
# ---------------- loop through files and fit ICA ------------------------
for file in files:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import preprocessed data --------------------------
    raw = read_raw_fif(file, preload=True)

    # sampling rate
    sfreq = raw.info['sfreq']

    # --- 3) set up ica parameters -----------------------------
    # ICA parameters
    n_components = 15
    method = 'picard'
    reject = dict(eeg=300e-6)

    # Pick electrodes to use
    picks = pick_types(raw.info,
                       eeg=True,
                       eog=False,
                       stim=False)

    # --- 4) fit ica --------------------------------------------
    # ICA parameters
    ica = ICA(n_components=n_components,
              method=method,
              fit_params=dict(ortho=False,
                              extended=True))

    # fit ICA
    ica.fit(raw.copy().filter(l_freq=1., h_freq=None),
            picks=picks,
            reject=reject,
            reject_by_annotation=True)

    # --- 5) save ica weights ----------------------------------
    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # save file
    ica.save(op.join(output_path, 'sub-%s' % subj,
                     'sub-%s_ica_weights-ica.fif' % subj))

    # --- 6) plot resulting components ------------------------
    # plot components
    ica_fig = ica.plot_components(picks=range(0, 15), show=False)
    ica_fig.savefig(op.join(output_path, 'sub-%s' % subj,
                            'sub-%s_ica.pdf' % subj))
