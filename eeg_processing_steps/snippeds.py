
# list comprehension
chans_to_drop = [chan for chan in raw.info['ch_names'] if (np.std(raw.get_data(raw.info['ch_names'].index(chan))) / 100e-6) < 1.]  # noqa

types = ['eeg' if chan in montage_chans else 'eog' if re.match('EOG|EXG', chan) else 'stim' for chan in chans]  # noqa


# preprocessing
eeg_picks = mne.pick_types(raw_blocks.info, eeg=True)

bad_channels = find_outliers(raw_blocks.get_data(eeg_picks))
