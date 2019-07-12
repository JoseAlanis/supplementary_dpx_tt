# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- extract epochs,
# --- export epoched data

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall
import numpy as np
import pandas as pd

from mne import events_from_annotations, pick_types, Epochs
from mne.io import read_raw_fif

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
data_path = op.join(derivatives_path, 'pruned_with_ica')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'epochs')):
    mkdir(op.join(derivatives_path, 'epochs'))

# path for saving output
output_path = op.join(derivatives_path, 'epochs')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# ========================================================================
# ------------- loop through files and extract epochs --------------------
for file in files:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # --- 2) import the data -----------------------------------
    raw = read_raw_fif(file, preload=True)

    # sampling frequency
    sfreq = raw.info['sfreq']

    # --- 3) create events and metadata for epoching -----------
    # get annotations
    annotations = pd.DataFrame(raw.annotations)
    # save latency of block boundary
    raw_break = annotations[annotations.description == 'EDGE boundary'].onset
    raw_break = float(raw_break)

    # --- recode cues ---
    #  get events
    events = events_from_annotations(raw)
    # copy of events
    new_evs = events[0].copy()

    # global variables fro recoding
    block = []
    probe_ids = []

    trial = 0

    reaction = []
    rt = []

    # loop trough events and recode
    for event in range(len(new_evs[:, 2])):
        # --- if event is a cue stimulus ---
        if new_evs[event, 2] in {5, 6, 7, 8, 9, 10}:

            # save block based on onset (before or after break)
            if (new_evs[event, 0] / sfreq) < raw_break:
                block.append(0)
            else:
                block.append(1)

            # --- 1st check: if next event is a false reaction ---
            if new_evs[event+1, 2] in {1, 2}:
                # if event is an A-cue
                if new_evs[event, 2] == 5:
                    # recode as too soon A-cue
                    new_evs[event, 2] = 18
                # if event is a B-cue
                elif new_evs[event, 2] in {6, 7, 8, 9, 10}:
                    # recode as too soon B-cue
                    new_evs[event, 2] = 19

                # look for next probe
                i = 2
                while new_evs[event+i, 2] not in {11, 12, 13, 14, 15, 16}:
                    i += 1

                # if probe is an X
                if new_evs[event+i, 2] == 11:
                    # recode as too soon X-probe
                    new_evs[event+i, 2] = 20
                # if probe is an Y
                else:
                    # recode as too soon Y-probe
                    new_evs[event+i, 2] = 21

                # save trial information as NaN
                trial += 1
                rt.append(np.nan)
                reaction.append(np.nan)
                # go on to next trial
                continue

            # --- 2nd check: if next event is a probe stimulus ---
            elif new_evs[event+1, 2] in {11, 12, 13, 14, 15, 16}:

                # if event after probe is a reaction
                if new_evs[event+2, 2] in {1, 2, 3, 4}:

                    # save reaction time
                    rt.append((new_evs[event+2, 0] - new_evs[event+1, 0]) / sfreq)

                    # if reaction is correct
                    if new_evs[event + 2, 2] in {3, 4}:

                        # save response
                        reaction.append(1)

                        # if cue was an A
                        if new_evs[event, 2] == 5:
                            # recode as correct A-cue
                            new_evs[event, 2] = 22

                            # if probe was an X
                            if new_evs[event+1, 2] == 11:
                                # recode as correct AX probe combination
                                new_evs[event+1, 2] = 23

                            # if probe was a Y
                            else:
                                # recode as correct AY probe combination
                                new_evs[event+1, 2] = 24

                            # go on to next trial
                            trial += 1
                            continue

                        # if cue was a B
                        else:
                            # recode as correct B-cue
                            new_evs[event, 2] = 25

                            # if probe was an X
                            if new_evs[event + 1, 2] == 11:
                                # recode as correct BX probe combination
                                new_evs[event + 1, 2] = 26
                            # if probe was a Y
                            else:
                                # recode as correct BY probe combination
                                new_evs[event + 1, 2] = 27

                            # go on to next trial
                            trial += 1
                            continue

                    # if reaction was incorrect
                    else:

                        # save response
                        reaction.append(0)

                        # if cue was an A
                        if new_evs[event, 2] == 5:
                            # recode as incorrect A-cue
                            new_evs[event, 2] = 28

                            # if probe was an X
                            if new_evs[event + 1, 2] == 11:
                                # recode as incorrect AX probe combination
                                new_evs[event + 1, 2] = 29

                            # if probe was a Y
                            else:
                                # recode as incorrect AY probe combination
                                new_evs[event + 1, 2] = 30

                            # go on to next trial
                            trial += 1
                            continue

                        # if cue was a B
                        else:
                            # recode as incorrect B-cue
                            new_evs[event, 2] = 31

                            # if probe was an X
                            if new_evs[event + 1, 2] == 11:
                                # recode as incorrect BX probe combination
                                new_evs[event + 1, 2] = 32

                            # if probe was a Y
                            else:
                                # recode as incorrect BY probe combination
                                new_evs[event + 1, 2] = 33

                            # go on to next trial
                            trial += 1
                            continue

                # if no reaction followed cue-probe combination
                elif new_evs[event + 2, 2] not in {1, 2, 3, 4}:

                    # save reaction time as NaN
                    rt.append(99999)
                    reaction.append(np.nan)

                    # if cue was an A
                    if new_evs[event, 2] == 5:
                        # recode as missed A-cue
                        new_evs[event, 2] = 34

                        # if probe was an X
                        if new_evs[event + 1, 2] == 11:
                            # recode as missed AX probe combination
                            new_evs[event + 1, 2] = 35

                        # if probe was a Y
                        else:
                            # recode as missed AY probe combination
                            new_evs[event + 1, 2] = 36

                        # go on to next trial
                        trial += 1
                        continue

                    # if cue was a B
                    else:
                        # recode as missed B-cue
                        new_evs[event, 2] = 37

                        # if probe was an X
                        if new_evs[event + 1, 2] == 11:
                            # recode as missed BX probe combination
                            new_evs[event + 1, 2] = 38

                        # if probe was a Y
                        else:
                            # recode as missed BY probe combination
                            new_evs[event + 1, 2] = 39

                        # go on to next trial
                        trial += 1
                        continue

        # skip other events
        else:
            continue

    # --- 4) set event ids -------------------------------------
    # cue events
    cue_event_id = {'Too_soon A': 18,
                    'Too_soon B': 19,

                    'Correct A': 22,
                    'Correct B': 25,

                    'Incorrect A': 28,
                    'Incorrect B': 31,

                    'Missed A': 34,
                    'Missed B': 37}

    # probe events
    probe_event_id = {'Too_soon X': 20,
                      'Too_soon Y': 21,

                      'Correct AX': 23,
                      'Correct AY': 24,

                      'Correct BX': 26,
                      'Correct BY': 27,

                      'Incorrect AX': 29,
                      'Incorrect AY': 30,

                      'Incorrect BX': 32,
                      'Incorrect BY': 33,

                      'Missed AX': 35,
                      'Missed AY': 36,

                      'Missed BX': 38,
                      'Missed BY': 39}

    # reversed event_id dict
    cue_event_id_rev = {val: key for key, val in cue_event_id.items()}
    probe_event_id_rev = {val: key for key, val in probe_event_id.items()}

    # --- 5) create metadata -----------------------------------
    # save cue events
    cue_events = new_evs[np.where((new_evs[:, 2] == 18) |
                                  (new_evs[:, 2] == 19) |
                                  (new_evs[:, 2] == 22) |
                                  (new_evs[:, 2] == 25) |
                                  (new_evs[:, 2] == 28) |
                                  (new_evs[:, 2] == 31) |
                                  (new_evs[:, 2] == 34) |
                                  (new_evs[:, 2] == 37))]

    # save probe events
    probe_events = new_evs[np.where((new_evs[:, 2] == 20) |
                                    (new_evs[:, 2] == 21) |
                                    (new_evs[:, 2] == 23) |
                                    (new_evs[:, 2] == 24) |
                                    (new_evs[:, 2] == 26) |
                                    (new_evs[:, 2] == 27) |
                                    (new_evs[:, 2] == 29) |
                                    (new_evs[:, 2] == 30) |
                                    (new_evs[:, 2] == 32) |
                                    (new_evs[:, 2] == 33) |
                                    (new_evs[:, 2] == 35) |
                                    (new_evs[:, 2] == 36) |
                                    (new_evs[:, 2] == 38) |
                                    (new_evs[:, 2] == 39))]

    # create list with reactions based on cue and probe event ids
    reaction_cues, reaction_probes, cues, probes = [], [], [], []
    for cue, probe in zip(cue_events[:, 2], probe_events[:, 2]):
        response, cue = cue_event_id_rev[cue].split(' ')
        reaction_cues.append(response)
        cues.append(cue)

        response, probe = probe_event_id_rev[probe].split(' ')
        reaction_probes.append(response)
        probes.append(probe)

    # create metadata
    metadata = {'block': block,
                'trial': range(0, trial),
                'cue': cues,
                'probe': probes,
                'reaction_cues': reaction_cues,
                'reaction_probes': reaction_probes,
                'rt': rt}
    # to data frame
    metadata = pd.DataFrame(metadata)

    # --- 6) extract epochs ------------------------------------
    # pick channels to keep
    picks = pick_types(raw.info, eeg=True)

    # rejection threshold
    reject = dict(eeg=300e-6)

    # create cue epochs
    cue_epochs = Epochs(raw, cue_events, cue_event_id,
                        metadata=metadata,
                        on_missing='ignore',
                        tmin=-2.,
                        tmax=2.5,
                        baseline=None,
                        preload=True,
                        reject_by_annotation=False,
                        picks=picks,
                        reject=reject)

    # create probe epochs
    probe_epochs = Epochs(raw, probe_events, probe_event_id,
                          metadata=metadata,
                          on_missing='ignore',
                          tmin=-2.,
                          tmax=2.,
                          baseline=None,
                          preload=True,
                          reject_by_annotation=False,
                          picks=picks,
                          reject=reject)

    # --- 7) save epochs info ------------------------------------
    # clean cue epochs
    clean_cues = cue_epochs.selection
    bad_cues = [x for x in set(list(range(0, trial)))
                if x not in set(cue_epochs.selection)]
    # clean probe epochs
    clean_probes = probe_epochs.selection
    bad_probes = [x for x in set(list(range(0, trial)))
                  if x not in set(probe_epochs.selection)]

    # --- 8) write summary ---------------------------------------

    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # write summary file
    name = op.join(output_path, 'sub-%s' % subj, 'sub-%s_epochs.txt' % subj)

    # open summary file
    sum_file = open(name, 'w')
    sum_file.write('clean_cue_epochs_are_' + str(len(clean_cues)) + '_:\n')
    for cue in clean_cues:
        sum_file.write('%s \n' % cue)

    sum_file.write('clean_probe_epochs_are_' + str(len(clean_probes)) + '_:\n')
    for probe in clean_probes:
        sum_file.write('%s \n' % probe)

    sum_file.write('clean_cue_epochs_are_' + str(len(bad_cues)) + ':\n')
    for bad_cue in bad_cues:
        sum_file.write('%s \n' % bad_cue)

    sum_file.write('clean_probe_epochs_are_' + str(len(bad_probes)) + ':\n')
    for bad_probe in bad_probes:
        sum_file.write('%s \n' % bad_probe)
    # Close summary file
    sum_file.close()

    # --- 9) save epochs ---------------------------------------
    cue_epochs.save(op.join(output_path,
                            'sub-%s' % subj,
                            'sub-%s_cues-epo.fif' % subj),
                    overwrite=True)
    probe_epochs.save(op.join(output_path,
                              'sub-%s' % subj,
                              'sub-%s_probes-epo.fif' % subj),
                      overwrite=True)
