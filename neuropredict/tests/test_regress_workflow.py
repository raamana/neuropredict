import os
import random
import shlex
import sys
from os.path import abspath, dirname, exists as pexists, join as pjoin, realpath

import numpy as np

sys.dont_write_bytecode = True

if __name__ == '__main__' and __package__ is None:
    parent_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(parent_dir)

from neuropredict.regress import cli
from pyradigm import RegressionDataset
from pyradigm.utils import (make_random_RegrDataset,
                            dataset_with_new_features_same_everything_else)

feat_generator = np.random.randn

random.seed(42)

test_dir = dirname(os.path.realpath(__file__))
out_dir = realpath(pjoin(test_dir, '..', 'tests', 'scratch_regress'))
if not pexists(out_dir):
    os.makedirs(out_dir)

min_size = 300
max_size = 500
max_dim = 100

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 20

covar_list = ('age', 'gender', 'dummy')
covar_types = ('age', 'gender', 'float')
covar_arg = ','.join(['age', ])

deconf_method = 'residualize'

red_dim = 'sqrt'
estimator = 'randomforestregressor'
dr_method = 'variancethreshold'  # 'selectkbest_f_classif'
gs_level = 'none'  # 'light'

num_procs = 1

out_path = os.path.join(out_dir, 'random_regr_ds1.pkl')
if pexists(out_path):
    ds_one = RegressionDataset(dataset_path=out_path)
else:
    ds_one = make_random_RegrDataset(min_size=min_size, max_size=max_size,
                                     max_dim=max_dim,
                                     attr_names=covar_list,
                                     attr_types=covar_types
                                     )
    ds_one.description = 'ds_one'
    ds_one.save(out_path)

out_path2 = os.path.join(out_dir, 'random_regr_ds2.pkl')
ds_two = dataset_with_new_features_same_everything_else(ds_one, max_dim)
ds_two.description = 'ds_two'
ds_two.save(out_path2)

sys.argv = shlex.split('np_regress -y {} {} -t {} -n {} -c {} -g {} -o {} '
                       '-e {} -dr {} -cl {} -cm {}'
                       ''.format(out_path, out_path2, train_perc, num_rep_cv,
                                 num_procs, gs_level, out_dir, estimator,
                                 dr_method, covar_arg, deconf_method))
cli()
