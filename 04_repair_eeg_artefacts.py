"""
===============================================
Repair EEG artefacts caused by ocular movements
===============================================

Identify "bad" components in ICA solution (e.g., components which are highly
correlated the time course of the electrooculogram).

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import read_ica, corrmap, create_eog_epochs

# All parameters are defined in config.py
from config import fname, parser

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Finding and removing bad components for subject %s' % subject)

###############################################################################
# 1) Import the output from previous processing step
input_file = fname.output(subject=subject,
                          processing_step='artefact_detection',
                          file_type='raw.fif')
raw = read_raw_fif(input_file, preload=True)

# apply average reference
raw_copy = raw.copy()
raw_copy.apply_proj()

###############################################################################
# 2) get EOG-channel indices and names

# get eogs indices and names
eogs = pick_types(raw_copy.info, eog=True)
eog_names = [raw.ch_names[ch] for ch in eogs]

if len(eog_names) > 2:
    raw_copy.drop_channels([eog_names[1], eog_names[2]])



###############################################################################
# 3) Import ICA weights from precious processing step
ica_file = fname.output(subject=subject,
                        processing_step='fit_ica',
                        file_type='ica.fif')
ica = read_ica(ica_file)

###############################################################################
raws = []
icas = []
# 4) Find "eog" components via correlation
for subject in subjects:
    input_file = fname.output(subject=subject,
                              processing_step='artefact_detection',
                              file_type='raw.fif')
    raws.append(read_raw_fif(input_file))

    ica_file = fname.output(subject=subject,
                            processing_step='fit_ica',
                            file_type='ica.fif')
    icas.append(read_ica(ica_file))

raw = raws[2]
ica = icas[2]

eog_indices, eog_scores = ica.find_bads_eog(raw,
                                            ch_name='EOGH_links',
                                            reject_by_annotation=True)

corrmap(icas, template=(2, 0), threshold=0.9, label='blink_up')
corrmap(icas, template=(1, 0), threshold=0.9, label='blink_weird',
        plot=False)
corrmap(icas, template=(2, 3), threshold=0.9, label='blink_side')


print([ica.labels_ for ica in icas])

icas[2].plot_components(picks=icas[2].labels_['blink_side'])

bad_components = []
bad_components.extend(icas[2].labels_['blink_up'])
bad_components.extend(icas[2].labels_['blink_side'])
bad_components.extend(icas[2].labels_['blink_weird'])

icas[2].exclude = bad_components

icas[2].plot_sources(raws[2])





# place holder for bad components
bad_comps = []

for n, eog in enumerate(eog_names[0:2]):
    eog_epochs = create_eog_epochs(raw_copy,
                                   ch_name=eog,
                                   reject_by_annotation=True)
    # create average blink
    eog_evoked = eog_epochs.average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))

    # find components that correlate with activity recorded at eog
    # channel in question
    eog_indices, eog_scores = ica.find_bads_eog(raw_copy,
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

