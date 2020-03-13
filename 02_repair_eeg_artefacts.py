"""
=============================================================
Extract segments of the data recorded during task performance
=============================================================

Segments that were recorded during the self-paced breaks (in between
experimental blocks) will be dropped.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
import matplotlib.pyplot as plt

from mne.io import read_raw_fif

# All parameters are defined in config.py
from config import fname, n_jobs, parser

# Handle command line arguments
args = parser.parse_args()
subject = args.subject

print('Converting subject %s to BIDS' % subject)
