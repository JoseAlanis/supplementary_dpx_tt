# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- compute results,
# --- save results objects

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall
import pickle

from mne import read_epochs

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
data_path = op.join(derivatives_path, 'epochs')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'all_epochs')):
    mkdir(op.join(derivatives_path, 'all_epochs'))

# path for saving output
output_path = op.join(derivatives_path, 'all_epochs')

# files to be analysed
cues_files = sorted(glob(op.join(data_path, 'sub-*', '*cues-epo.fif')))
probes_files = sorted(glob(op.join(data_path, 'sub-*', '*probes-epo.fif')))

# --- global variables to store results ---
all_cues = dict()
all_probes = dict()

# ========================================================================
# ------------- loop through files and append epochs ---------------------

# --- 1) extract cues ------------------------------------------
for file in cues_files:

    # set up paths and file names
    filepath, filename = op.split(file)

    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # import cue epochs
    cue_epochs = read_epochs(file, preload=True)

    # # resample down to improve computation time (uncomment if needed)
    cue_epochs.resample(100., npad='auto')

    # store cue epochs in dict
    all_cues[subj] = cue_epochs

# --- 2) extract probes ----------------------------------------
for file in probes_files:

    # set up paths and file names
    filepath, filename = op.split(file)

    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # import cue epochs
    probe_epochs = read_epochs(file, preload=True)

    # # resample down to improve computation time (uncomment if needed)
    probe_epochs.resample(100., npad='auto')

    # store probe epochs in dict
    all_probes[subj] = probe_epochs

# --- 3) save results ------------------------------------------
# save cues
with open(op.join(output_path, 'all_cue_epochs.pkl'), 'wb') as cues:
    pickle.dump(all_cues, cues)

# save probes
with open(op.join(output_path, 'all_probe_epochs.pkl'), 'wb') as probes:
    pickle.dump(all_probes, probes)
