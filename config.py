"""
===========
Study configuration file
===========

Configuration parameters and global variable values for the study.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import os
import getpass
from socket import getfqdn

import numpy as np

from utils import FileNames

###############################################################################
# Determine which user is running the scripts on which machine. Set the path to
# where the data is stored and determine how many CPUs to use for analysis.

user = getpass.getuser()  # Username
host = getfqdn()  # Hostname

# You want to add your machine to this list
if user == 'josealanis' and host == 'mwla514.wlan.uni-marburg.de':
    # iMac at work
    data_dir = '../data'
    n_jobs = 2  # iMac has 4 cores (we'll use 2).
# elif user == 'josealanis' and host == 'josealanis-desktop':
#     # pc at home
#     raw_data_dir = './data'
#     n_jobs = 16  # My workstation has 16 cores (we'll use 8).
else:
    # Defaults
    data_dir = './data'
    n_jobs = 1

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)

###############################################################################
# Relevant parameters for the analysis.
sample_rate = 256.  # Hz
fmin = 1.0  # Hz
fmax = 20.0  # Hz

# subjects to use for analysis
subjects = np.arange(1, 53)

###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# directories to use for input and output
fname.add('data_dir', data_dir)
fname.add('sourcedata_dir',
          '{data_dir}/sourcedata')
fname.add('derivatives_dir',
          '{data_dir}/derivatives')

# The data files that are used and produced by the analysis steps
fname.add('source',
          '{sourcedata_dir}/sub-{subject:02d}/sub-{subject:02d}.bdf')
fname.add('derivatives',
          '{derivatives_dir}/{processing_step}/sub-{subject:02d}')
fname.add('output',
          '{derivatives}/sub-{subject:02d}-{processing_step}-{file_type}.fif')

# Filenames for MNE reports
fname.add('report',
          '{derivatives}/sub-{subject:02d}-{processing_step}-report.h5')
fname.add('report_html',
          '{derivatives}/sub-{subject:02d}-{processing_step}-report.html')
