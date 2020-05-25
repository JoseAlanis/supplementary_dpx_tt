"""
This script validates the "sourcedata/" directory provided by user to
see if it's contents are consistent with the BIDS specification.
"""
import os

import re

import warnings

from config import fname

path = fname.sourcedata_dir


def validate_sourcedata(path):

    # if path is None:
    #     path = './'
    #     warnings.warn('No specific path to "sourcedata/" was provided. '
    #                   'Looking for "sourcedata/" in %s ' % path)
    with open(fname.validator, 'w') as f:
        f.write('System check OK.')

    dirs = os.listdir(path)

    if 'sourcedata' not in dirs:
        raise ValueError('The directory "%s" does not contain a '
                         '"sourcedata/" dirctory. '
                         'Have you chosen the right folder? '
                         'If so, check for typos the directory names. '
                         'You must provide a parent directory containing '
                         'a "sourcedata/" folder. The latter in turn, '
                         'should contain the data in individual subject '
                         'folders (e.g., '
                         './sourcedata/sub-01/eeg/sub-01.xdf)' % path)

    sourcedata_dir = os.path.join(path, 'sourcedata')

    dir_tree = [os.listdir(dr[0]) for dr in os.walk(sourcedata_dir)]

    if not dir_tree:
        warnings.warn('"./sourcedata/" appears to be empty. '
                      'Terminating validation process.')
        return

    for ib, branch in enumerate(dir_tree):
        if ib == 0:
            subj_dirs = [os.path.join(sourcedata_dir, sub)
                         for sub in branch
                         if os.path.isdir(os.path.join(sourcedata_dir, sub))]
            n_subjects = len(subj_dirs)
            print('Found %d folders in "%s"' % (n_subjects, sourcedata_dir))


            for s_dir in subj_dirs:
                if re.findall(pattern=r'sub-\d+',
                              string=os.path.basename(s_dir)):

