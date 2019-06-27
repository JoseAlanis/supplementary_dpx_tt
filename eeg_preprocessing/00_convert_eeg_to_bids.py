# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- import data,
# --- convert to bids format

# ========================================================================
# ------------------- import relevant extensions -------------------------
import glob
import os

import re
import numpy as np

import mne
from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

# ========================================================================
# --- global settings
# --- prompt user to set project path
root_path = input("Type path to project directory: ")

# look for directory
if os.path.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# path to eeg files
data_path = os.path.join(root_path, 'data/sourcedata/')

output_path = os.path.join(root_path, 'data_bids/')
# # path for output
# derivatives_path = os.path.join(root_path, 'derivatives')
#
# # create directory for save
# if not os.path.isdir(derivatives_path):
#     os.mkdir(derivatives_path)
#     os.mkdir(os.path.join(derivatives_path, 'extract_blocks'))
#
# output_path = os.path.join(derivatives_path, 'extract_blocks')
#
# files to be analysed
files = glob.glob(os.path.join(data_path, 'sub-*', 'eeg/*.bdf'))

# ========================================================================
# -- define further variables that apply to all files in the data set
task = 'dpxtt'
task_description = 'DPX, effects of time on task'
# eeg channel names and locations
montage = mne.channels.read_montage(kind='biosemi64')

# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8']

# ========================================================================
# ------------ loop through files and extract blocks  --------------------
for file in files:

    # --- 1) set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the data -----------------------------------
    raw = mne.io.read_raw_bdf(file,
                              montage=montage,
                              preload=False,
                              exclude=exclude)

    # reset `orig_time` in annotations
    raw.annotations.orig_time = None

    # Get events
    events = mne.find_events(raw,
                             stim_channel='Status',
                             output='onset',
                             min_duration=0.002)
    # event ids
    events_id = {'correct_target_button': 13,
                 'correct_non_target_button': 12,
                 'incorrect_target_button': 113,
                 'incorrect_non_target_button': 112,
                 'a_cue': 70,
                 'b_cue': range(71, 76),
                 'x_probe': 76,
                 'y_probe': range(77, 82)}

    # save in bids format
    bids_basename = make_bids_basename(subject=subj, task=task)

    write_raw_bids(raw, bids_basename, output_path, event_id=events_id,
                   events_data=events, overwrite=True)
