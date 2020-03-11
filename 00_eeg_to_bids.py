"""
========================================================
Bring data set into a BIDS compliant directory structure
========================================================

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os.path as op

from datetime import datetime
import pandas as pd

from mne.io import read_raw_bdf
from mne import find_events, Annotations, open_report

from mne_bids import write_raw_bids, make_bids_basename

# All parameters are defined in config.py
from config import fname, exclude, task_name, montage, parser

###############################################################################
# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Converting subject %s to BIDS' % subject)

###############################################################################
# Subject information (e.g., age, sex)
subj_demo = pd.read_csv(fname.subject_demographics, sep='\t', header=0)

###############################################################################
input_file = fname.source(subject=subject)
# 1) import the data
raw = read_raw_bdf(input_file,
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
    elif chan.startswith('EOG') | chan.startswith('EXG'):
        types.append('eog')
    else:
        types.append('stim')

# add channel types and eeg-montage
raw.set_channel_types({chan: typ for chan, typ in zip(channels, types)})
raw.set_montage(montage)

# compute approx. date of birth
# get measurement date from dataset info
date_of_record = raw.info['meas_date']
# convert to date format
date = datetime.utcfromtimestamp(date_of_record[0]).strftime('%Y-%m-%d')

# here, we compute only and approximate of the subjects birthday
# this is to keep the date anonymous (at least to some degree)
age = subj_demo[subj_demo.subject_id == 'sub-' + str(subject).rjust(3, '0')].age
sex = subj_demo[subj_demo.subject_id == 'sub-' + str(subject).rjust(3, '0')].sex

year_of_birth = int(date.split('-')[0]) - int(age)
approx_birthday = (year_of_birth,
                   int(date[5:].split('-')[0]),
                   int(date[5:].split('-')[1]))

# add modified subject info to dataset
raw.info['subject_info'] = dict(id=subject,
                                sex=int(sex),
                                birthday=approx_birthday)

# frequency of power line
raw.info['line_freq'] = 50.0

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

###############################################################################
# 6) Extract events from the status channel and save them as file annotations
# events to data frame
events = pd.DataFrame(events,
                      columns=['onset', 'duration', 'description'])
# onset to seconds
events['onset_in_s'] = events['onset'] / raw.info['sfreq']
# sort by onset
events = events.sort_values(by=['onset_in_s'])
# only keep relevant events
events = events.loc[(events['description'] <= 245)]

# crate annotations object
annotations = Annotations(events['onset_in_s'],
                          events['duration'],
                          events['description'],
                          orig_time=raw.info['meas_date'])
# apply to raw data
raw.set_annotations(annotations)

###############################################################################
# 7) Plot the data for report
raw_plot = raw.plot(scalings=dict(eeg=50e-6, eog=50e-6),
                    n_channels=len(raw.info['ch_names']),
                    show=False)

###############################################################################
# 8) export data to .fif for further processing
# output path
output_path = fname.output(processing_step='raw_files',
                           subject=subject,
                           file_type='raw.fif')

# save file
raw.save(output_path, overwrite=True)

###############################################################################
# 9) create HTML report
with open_report(fname.report(subject=subject)[0]) as report:
    report.parse_folder(op.dirname(output_path), pattern='*.fif')
    report.add_figs_to_section(raw_plot, 'Raw data', section='raw_data',
                               replace=True)
    report.save(fname.report(subject=subject)[1], overwrite=True,
                open_browser=False)
