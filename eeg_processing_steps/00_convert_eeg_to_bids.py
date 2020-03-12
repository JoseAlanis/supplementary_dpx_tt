"""
========================================================
Bring data set into BIDS a compliant directory structure
========================================================

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)

Notes
------
Versions: Python 3.7; mne 0.19.2

"""
import glob
import os
import re
from datetime import datetime

from pandas import read_csv

from mne.channels import make_standard_montage
from mne.io import read_raw_bdf
from mne import create_info, find_events

from mne_bids import write_raw_bids, make_bids_basename

# ========================================================================
# global settings
root_path = input("Type path to project directory: ")  # prompt to set path

# look for directory
if os.path.isdir(root_path):
    print("Setting 'root_path' to ", root_path)
else:
    raise NameError('Directory not found!')

# input path
data_path = os.path.join(root_path, 'sourcedata/')

# path to subject demographics
subj_demo = os.path.join(root_path, 'subject_data/subject_demographics.tsv')

# files to be analysed
files = sorted(glob.glob(os.path.join(data_path, 'sub-*', '*.bdf')))

# subjects data
subj_demo = read_csv(subj_demo, sep='\t', header=0)

# define further variables that apply to all files in the data set
task = 'dpxtt'
task_description = 'DPX, effects of time on task'
# eeg channel names and locations
montage = make_standard_montage(kind='standard_1020')
# channels in eeg-montage
montage_channels = montage.ch_names

# channels to be exclude from import
exclude = ['EXG5', 'EXG6', 'EXG7', 'EXG8']

# ========================================================================
# --------- loop through files and make bids-files blocks  ---------------
for ind, file in enumerate(files):

    # --- 1) set up paths and file names -----------------------
    filepath, filename = os.path.split(file)
    # subject in question
    subj = re.findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the data -----------------------------------
    raw = read_raw_bdf(file,
                       preload=False,
                       exclude=exclude)

    # sampling rate
    sfreq = raw.info['sfreq']
    # channels in dataset
    channels = raw.info['ch_names']

    # --- 3) modify dataset info -------------------------------
    # identify channel types based on matching names in montage
    types = []
    for chan in channels:
        if chan in montage_channels:
            types.append('eeg')
        elif re.match('EOG|EXG', chan):
            types.append('eog')
        else:
            types.append('stim')

    # number channels of type 'eeg' in dataset
    n_eeg = len([chan for chan in channels if chan in montage_channels])

    # create custom info for subj file
    info_custom = create_info(channels, sfreq, types, montage)

    # --- 4) compute approx. date of birth ---------------------
    # get measurement date from dataset info
    date_of_record = raw.info['meas_date'][0]
    # convert to date format
    date = datetime.utcfromtimestamp(date_of_record).strftime('%Y-%m-%d')

    # here, we compute only and approximate of the subjects birthday
    # this is to keep the date anonymous (at least to some degree)
    year_of_birth = int(date.split('-')[0]) - subj_demo.iloc[ind].age
    approx_birthday = (year_of_birth,
                       int(date[5:].split('-')[0]),
                       int(date[5:].split('-')[1]))

    # add modified subject info to dataset
    raw.info['subject_info'] = dict(id=int(subj),
                                    sex=subj_demo.iloc[ind].sex,
                                    birthday=approx_birthday)

    # --- 4) create events info --------------------------------
    # extract events
    events = find_events(raw,
                         stim_channel='Status',
                         output='onset',
                         min_duration=0.002)

    # event ids
    events_id = {'correct_target_button': 13,
                 'correct_non_target_button': 12,
                 'incorrect_target_button': 113,
                 'incorrect_non_target_button': 112,
                 'cue_0': 70,
                 'cue_1': 71,
                 'cue_2': 72,
                 'cue_3': 73,
                 'cue_4': 74,
                 'cue_5': 75,
                 'probe_0': 76,
                 'probe_1': 77,
                 'probe_2': 78,
                 'probe_3': 79,
                 'probe_4': 80,
                 'probe_5': 81,
                 'start_record': 127,
                 'pause_record': 245}

    # --- 5) export to bids ------------------------------------
    # file name compliant with bids
    bids_basename = make_bids_basename(
        subject=subj,
        task=task)

    # save in bids format
    write_raw_bids(raw,
                   bids_basename,
                   root_path,
                   event_id=events_id,
                   events_data=events,
                   overwrite=True)