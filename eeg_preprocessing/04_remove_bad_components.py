# --- jose C. garcia alanis
# --- utf-8
# --- Python 3.7.3 / mne 0.18.1
#
# --- eeg pre-processing for DPX TT
# --- version: june 2019
#
# --- remove bad components,
# --- export pruned data

# ========================================================================
# ------------------- import relevant extensions -------------------------
import os.path as op
from os import mkdir
from glob import glob

from re import findall

from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import read_ica, create_eog_epochs

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
# path to ica files
ica_path = op.join(derivatives_path, 'ica')

# create directory for output
if not op.isdir(op.join(derivatives_path, 'pruned_with_ica')):
    mkdir(op.join(derivatives_path, 'pruned_with_ica'))

# path for saving output
output_path = op.join(derivatives_path, 'pruned_with_ica')

# files to be analysed
files = sorted(glob(op.join(data_path, 'sub-*', '*-raw.fif')))

# ========================================================================
# ------------ loop through files and remove bad components --------------
for file in files[30:]:

    # --- 1) Set up paths and file names -----------------------
    filepath, filename = op.split(file)
    # subject in question
    subj = findall(r'\d+', filename)[0].rjust(3, '0')

    # create directory for save
    if not op.exists(op.join(output_path, 'sub-%s' % subj)):
        mkdir(op.join(output_path, 'sub-%s' % subj))

    # --- 2) import the data -----------------------------------
    raw = read_raw_fif(file, preload=True)

    # apply average reference
    raw.apply_proj()

    # get eogs indices and names
    eogs = pick_types(raw.info, eog=True)
    eog_names = [raw.ch_names[ch] for ch in eogs]

    # if len(eogs) > 2:
    #     raw.drop_channels(raw.info['ch_names'][-3])
    #     raw.drop_channels(raw.info['ch_names'][-1])

    # --- 4) import ICA weights --------------------------------
    ica = read_ica(op.join(ica_path,
                           'sub-%s' % subj,
                           'sub-%s_ica_weights-ica.fif' % subj))

    # --- 5) create average blinks and save figure -------------
    # plotting parameters
    ts_args = dict(ylim=dict(eeg=[-25, 125]))
    topomap_args = dict(vmax=100, vmin=-25)

    # place holder for bad components
    bad_comps = []

    # --- 6) find "eog" components via correlation -------------
    for n, eog in enumerate(eog_names):
        eog_epochs = create_eog_epochs(raw,
                                       reject_by_annotation=True,
                                       # picks='eeg',
                                       ch_name=eog)
        # create average blink
        eog_evoked = eog_epochs.average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))

        # find components that correlate with activity recorded at eog
        # channel in question
        eog_indices, eog_scores = ica.find_bads_eog(eog_epochs,
                                                    threshold=3.0,
                                                    ch_name=eog,
                                                    reject_by_annotation=True)

        for eog_i in eog_indices:
            bad_comps.append(eog_i)
            # plot component properties
            fig = ica.plot_properties(eog_epochs,
                                      picks=eog_i,
                                      psd_args={'fmax': 35.},
                                      image_args={'sigma': 1.})[0]
            fig.savefig(op.join(output_path, 'sub-%s' % subj,
                                'sub-%s_comp_%d.pdf' % (subj, eog_i)))

        fig = ica.plot_scores(eog_scores,
                              exclude=eog_indices,
                              title='scores %s' % eog)
        fig.savefig(op.join(output_path, 'sub-%s' % subj,
                            'sub-%s_r-%s_scores.pdf' % (subj, eog)))

    # --- 7) Exclude bad components ----------------------------
    ica.exclude = list(set(bad_comps))

    # check if any others should be removed
    ica.plot_sources(raw, block=True)
    print(ica.exclude)

    # --- 8) remove bad components -----------------------------
    # apply ica weights to raw data
    ica.apply(raw)

    # --- 7) check results -------------------------------------
    # plot pruned data
    raw.plot(n_channels=len(raw.ch_names),
             title=str(filename),
             scalings=dict(eeg=50e-6),
             bad_color='red',
             block=True)

    # --- 8) save pruned data -----------------------------------
    # pick electrodes to save
    picks = pick_types(raw.info,
                       eeg=True,
                       eog=False)

    # save file
    raw.save(op.join(output_path, 'sub-%s' % subj,
                     'sub-%s_pruned-raw.fif' % subj),
             overwrite=True)
