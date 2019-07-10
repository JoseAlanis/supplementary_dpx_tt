# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- ica decomposition, find eog artifacts,
# --- export ica solution

# =================================================================================================
# ------------------------------ Import relevant extensions ---------------------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall

import numpy as np
import pandas as pd

from mne import find_events, Annotations
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
data_path = op.join(derivatives_path, 'extract_blocks')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'ica')):
    mkdir(op.join(derivatives_path, 'ica'))

# path for saving output
output_path = op.join(derivatives_path, 'ica')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# === LOOP THROUGH FILES AND RUN PRE-PROCESSING ==========================
for file in files:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) READ IN THE DATA ----------------------------------
    # Import preprocessed data.
    raw = read_raw_fif(file, preload=True)

    # sampling rate
    sfreq = raw.info['sfreq']

    # annotations in data
    annot_infos = ['onset', 'duration', 'description']
    annotations = pd.DataFrame(raw.annotations)
    annotations = annotations[annot_infos]

    # --- 3) GET EVENT INFORMATION -----------------------------
    # Get events
    events = find_events(raw,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)
    # events as data frame
    events = pd.DataFrame(events,
                          columns=annot_infos)

    # onsets to seconds
    events.onset = events.onset / sfreq

    events = events.append(annotations, ignore_index=True)
    events = events.sort_values(by=['onset'])

    annotations = Annotations(events['onset'],
                              events['duration'],
                              events['description'],
                              orig_time=date_of_record)

    raw.set_annotations(annotations)

    raw_copy.plot(n_channels=67, scalings=dict(eeg=1e-4))

    # --- 2) ICA DECOMPOSITION --------------------------------
    # ICA parameters
    n_components = 25
    method = 'extended-infomax'
    # decim = None
    reject = dict(eeg=4e-4)

    # Pick electrodes to use
    picks = mne.pick_types(raw.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False)

    # ICA parameters
    ica = ICA(n_components=n_components,
              method=method)

    # Fit ICA
    ica.fit(raw.copy().filter(1, 50),
            picks=picks,
            reject=reject)

    ica.save(ica_path + filename.split('-')[0] + '-ica.fif')

    # --- 3) PLOT RESULTING COMPONENTS ------------------------
    # Plot components
    ica_fig = ica.plot_components(picks=range(0, 25), show=False)
    ica_fig.savefig(summary_path + filename.split('-')[0] + '_ica.pdf')
