from os.path import abspath, dirname, exists as pexists, join as pjoin, realpath
import os
import shlex
import sys
from neuropredict import cli

in_dir = '/Volumes/data/work/rotman/CANBIND/data/Tier1_v1/processing'
out_dir = pjoin(in_dir, 'missing')
os.makedirs(out_dir, exist_ok=True)

ds_with_missing = pjoin(out_dir, 'CANBIND_Tier1_Clinical_features.KeepNaN.MLDataset.pkl')

sys.argv = shlex.split('neuropredict -n 10 -t 0.8 -fs selectkbest_mutual_info_classif '
                       '-e svm -y {} -o {}'.format(ds_with_missing, out_dir))

cli()