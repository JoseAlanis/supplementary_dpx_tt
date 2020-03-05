"""
========================================================
Bring data set into a BIDS compliant directory structure
========================================================

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import argparse
import re
from datetime import datetime

from pandas import read_csv

from mne.io import read_raw_bdf
from mne import create_info, find_events

from mne_bids import write_raw_bids, make_bids_basename

# All parameters are defined in config.py
from config import fname, exclude, task_name, montage

###############################################################################
# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject

print('Converting subject %s to BIDS' % subject)

###############################################################################
# Subject information (e.g., age, sex)
subj_demo = read_csv(fname.subject_demographics, sep='\t', header=0)

###############################################################################
# 1) import the data
raw = read_raw_bdf(fname.source(subject=int(subject)),
                   preload=False,
                   exclude=exclude)

# sampling rate
sfreq = raw.info['sfreq']
# channels in dataset
channels = raw.info['ch_names']

###############################################################################
# 2) modify dataset info
# identify channel types based on matching names in montage
types = []
for chan in channels:
    if chan in montage.ch_names:
        types.append('eeg')
    elif re.match('EOG|EXG', chan):
        types.append('eog')
    else:
        types.append('stim')

# number channels of type 'eeg' in dataset
n_eeg = len([chan for chan in channels if chan in montage.ch_names])

# create custom info for subj file
info_custom = create_info(channels, sfreq, types, montage)

###############################################################################
# 3) compute approx. date of birth
# get measurement date from dataset info
date_of_record = raw.info['meas_date'][0]
# convert to date format
date = datetime.utcfromtimestamp(date_of_record).strftime('%Y-%m-%d')

# here, we compute only and approximate of the subjects birthday
# this is to keep the date anonymous (at least to some degree)
age = subj_demo[subj_demo.subject_id == 'sub-' + subject.rjust(3, '0')].age
sex = subj_demo[subj_demo.subject_id == 'sub-' + subject.rjust(3, '0')].sex

year_of_birth = int(date.split('-')[0]) - int(age)
approx_birthday = (year_of_birth,
                   int(date[5:].split('-')[0]),
                   int(date[5:].split('-')[1]))

# add modified subject info to dataset
raw.info['subject_info'] = dict(id=int(subject),
                                sex=int(sex),
                                birthday=approx_birthday)

###############################################################################
# 4) create events info
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

###############################################################################
# 5) export to bids
# file name compliant with bids
bids_basename = make_bids_basename(
    subject=str(subject).rjust(3, '0'),
    task=task_name)

# save in bids format
write_raw_bids(raw,
               bids_basename,
               fname.data_dir,
               event_id=events_id,
               events_data=events,
               overwrite=True)
