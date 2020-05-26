"""
This script validates the "sourcedata/" directory provided by user to
see if it's contents are consistent with the BIDS specification.
"""
import os

import re

import warnings

import numpy as np

from config import fname, parser

# Handle command line arguments
args = parser.parse_args()
parent_dir = args.path


def validate_sourcedata(path, source_type, pattern='sub-\\d+'):

    if not path:
        try:
            path = fname.data_dir
        except ValueError:
            print('No valid data path found. The validator ecountered an error '
                  'while looking for an appropiate parent directory in '
                  '"./config.py". please provide an appropriate '
                  'path to look for "sourcedata/"')

    if not source_type:
        source_type = ['eeg']

    # construct relative sourcedata path
    sourcedata_dir = os.path.join(path, 'sourcedata')

    source_name = None

    sub_dirs = []
    n_subs = None
    subject_names = None

    data_dirs = []

    # browse through directories in parent_dir
    for root, directories, files in os.walk(path):
        if root == path:
            if 'sourcedata' not in directories:
                raise ValueError('The directory "%s" does not contain a '
                                 '"sourcedata/" directory. '
                                 'Have you chosen the right folder? '
                                 'If so, check for typos the directory names. '
                                 'You must provide a parent directory '
                                 'containing a "sourcedata/" folder. '
                                 'The latter in turn, should contain the data '
                                 'in individual subject folders (e.g., '
                                 './sourcedata/sub-01/eeg/sub-01.xdf)' % path)
            else:
                print('Found "sourcedata/" in %s' % path)
                source_name = 'OK'

        if root == sourcedata_dir:
            sub_dirs.extend([os.path.join(root, sub) for sub in directories])
            n_subs = len(sub_dirs)
            print('Found %d folders in "%s"' % (n_subs, sourcedata_dir))

            valid_names = []
            for sub in sub_dirs:
                if re.findall(pattern=pattern,
                              string=os.path.basename(sub)):
                    valid_names.append(True)
                else:
                    valid_names.append(False)

            if len(set(valid_names)) > 1:
                bad_names = np.where(np.logical_not(valid_names))[0]
                bad_names = [sub_dirs[bn] for bn in bad_names]

                warnings.warn('The directory names in "%s" '
                              'are not BIDS conform. '
                              'Directory names in sourcedata/ should follow '
                              'the patter: "./sub-<subject-label>/" '
                              '(consider checking for upper/lower-case '
                              'inconsistencies).'
                              % (', '.join([str(bn) for bn in bad_names])))
                subject_names = 'error in %s' \
                                % (', '.join([str(bn) for bn in bad_names]))
            else:
                subject_names = 'all OK'

        if root in sub_dirs:
            data_dirs.append([data_dir for data_dir in directories
                              if data_dir in source_type])
            n_dirs = [len(d) for d in data_dirs]
            if len(set(n_dirs)) > 1:
                warnings.warn('The subject directories have inconsistent '
                              'number of sub directories. Please consider '
                              'checking for missing or misspellings in '
                              'the directory names.')
                modal = max(set(n_dirs), key=n_dirs.count)

                inconscistent = [i for i, n in enumerate(n_dirs) if not n == modal]
                bads = [sub_dirs[i] for i in inconscistent]


    data = {
        'source_data_path': {
            'path': sourcedata_dir,
            'naming': source_name,
            'dirs_in_sourcedata': n_subs
        },
        'subject_directories':
            {
                'name_pattern': pattern,
                'naming': subject_names
            }
    }

