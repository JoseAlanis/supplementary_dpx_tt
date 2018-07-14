
for i in range(new_events[:, 2].size):
    # FIRST STEP: temp_cue == 0 indicates we are looking for 'cue stimuli'.
    # (i.e., events in {70, 71, 72, 73, 74, 75}.
    if temp_cue == 0:
        # event 70 is an 'A cue'
        if new_events[:, 2][i] == 70:
            # 'A cue' was found; move on
            temp_cue = 1
        # events 71, 72, 73, 74, 75 are 'B cues'.
        elif new_events[:, 2][i] in {71, 72, 73, 74, 75}:
            # 'B cue' was found; move on
            temp_cue = 2
        continue
    # SECOND STEP: look for 'probe stimuli' (temp_cue > 0)
    elif temp_cue == 1:
        # cues followed by wrong key presses (events 112 & 113)
        # should be marked as invalid
        if new_events[:, 2][i] in {112, 113}:
            valid = False
            continue
        if valid is True:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 1
                # Set the temp_cue back to 0.
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 3
                # Set the temp_cue back to 0.
            temp_cue = 0
        elif valid is False:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 11
                # Set the temp_cue back to 0.
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 31
                # Set the temp_cue back to 0.
            temp_cue = 0
            valid = True
    elif temp_cue == 2:
        # cues followed by wrong key presses (events 112 & 113)
        # should be marked as invalid
        if new_events[:, 2][i] in {112, 113}:
            valid = False
            continue
        if valid is True:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 2
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:\
                new_events[:, 2][i] = 4
            temp_cue = 0
        elif valid is False:
            if new_events[:, 2][i] == 76:
                new_events[:, 2][i] = 21
            elif new_events[:, 2][i] in {77, 78, 79, 80, 81}:
                new_events[:, 2][i] = 41
            temp_cue = 0
            valid = True