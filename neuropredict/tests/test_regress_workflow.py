import os
import shlex
import sys
from os.path import abspath, dirname, exists as pexists, join as pjoin, realpath
from sys import version_info

import numpy as np

sys.dont_write_bytecode = True

from pytest import raises

if __name__ == '__main__' and __package__ is None:
    parent_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(parent_dir)

from neuropredict import config_neuropredict as cfg
from neuropredict.regress import cli
from pyradigm import RegressionDataset
from pyradigm.utils import make_random_RegrDataset

feat_generator = np.random.randn

test_dir = dirname(os.path.realpath(__file__))
out_dir = realpath(pjoin(test_dir, '..', 'tests', 'scratch_regress'))
if not pexists(out_dir):
    os.makedirs(out_dir)

min_size=300
max_size=500
max_dim = 100

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 50
red_dim = 'sqrt'
estimator = 'randomforestregressor'
dr_method = 'variancethreshold' # 'selectkbest_f_classif'
gs_level = 'none' # 'light'

num_procs = 1


def new_dataset_with_same_ids_targets(in_ds):
    feat_dim = np.random.randint(1, max_dim)
    out_ds = RegressionDataset()
    for id_ in in_ds.samplet_ids:
        out_ds.add_samplet(id_,  np.random.rand(feat_dim), target=in_ds.targets[id_])
    return out_ds

out_path = os.path.join(out_dir, 'random_regr_ds1.pkl')
ds_one = make_random_RegrDataset(min_size=min_size, max_size=max_size,
                                 max_dim=max_dim)
ds_one.save(out_path)

out_path2 = os.path.join(out_dir, 'random_regr_ds2.pkl')
ds_two = new_dataset_with_same_ids_targets(ds_one)
ds_two.save(out_path2)

sys.argv = shlex.split('np_regress -y {} {} -t {} -n {} -c {} -g {} -o {} '
                       '-e {} -dr {}'
                       ''.format(out_path, out_path2, train_perc, num_rep_cv,
                                 num_procs, gs_level, out_dir, estimator, dr_method))
cli()
