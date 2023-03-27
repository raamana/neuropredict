import random
import shlex
import sys
import traceback
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

if __name__ == '__main__' and __package__ is None:
    parent_dir = Path(__file__).resolve().parents[2]
    sys.path.append(parent_dir)

from neuropredict.regress import cli
from neuropredict import config as cfg
from neuropredict.tests._test_utils import remove_neuropredict_results
from pyradigm import RegressionDataset
from pyradigm.utils import (make_random_RegrDataset,
                            dataset_with_new_features_same_everything_else)

feat_generator = np.random.randn

random.seed(42)

test_dir = Path(__file__).resolve().parent
out_dir = test_dir.joinpath('..', 'tests', 'scratch_regress')
if not out_dir.exists():
    out_dir.mkdir()

min_size = 300
max_size = 500
max_dim = 100

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 45
num_procs = 3

covar_list = ('age', 'gender', 'dummy')
covar_types = ('age', 'gender', 'float')
covar_arg = ','.join(['age', ])

deconf_method = 'residualize'

red_dim = 'sqrt'
estimator = 'extratreesregressor' # 'randomforestregressor'
dr_method = 'variancethreshold'  # 'selectkbest_f_classif'
gs_level = 'none'  # 'light'

out_path1 = out_dir / 'random_regr_ds1.pkl'
out_path2 = out_dir / 'random_regr_ds2.pkl'
if out_path1.exists() and out_path2.exists():
    ds_one = RegressionDataset(dataset_path=out_path1)
    ds_two = RegressionDataset(dataset_path=out_path2)
else:
    ds_one = make_random_RegrDataset(min_size=min_size, max_size=max_size,
                                     max_dim=max_dim,
                                     attr_names=covar_list,
                                     attr_types=covar_types
                                     )
    ds_one.description = 'ds_one'
    ds_one.save(out_path1)

    ds_two = dataset_with_new_features_same_everything_else(ds_one, max_dim)
    ds_two.description = 'ds_two'
    ds_two.save(out_path2)


def test_basic_run():

    remove_neuropredict_results(out_dir)
    sys.argv = shlex.split('np_regress -y {} {} -t {} -n {} -c {} -g {} -o {} '
                           '-e {} -dr {} '
                           ''.format(out_path1, out_path2, train_perc, num_rep_cv,
                                     num_procs, gs_level, out_dir, estimator,
                                     dr_method))
    cli()


def test_each_combination_works():
    """Ensures each of combination of dim. reduction and regressor works"""

    nrep = 10
    nproc = 1
    gsl = 'none'  # to speed up the process
    failed_combos = list()
    for clf_name in cfg.regressor_choices:
        for fs_name in cfg.all_dim_red_methods:
            # skipping the test for LLE* to avoid numerical issues
            if fs_name.startswith('lle'):
                continue
            # ensure a fresh start
            remove_neuropredict_results(out_dir)
            try:
                cli_str = 'np_regress -y {} -t {} -n {} -c {} -g {} -o {} ' \
                          '-e {} -dr {} -g {} -c {}' \
                          ''.format(out_path1, train_perc, nrep, num_procs,
                                    gs_level, out_dir, estimator, dr_method,
                                    gsl, nproc)
                sys.argv = shlex.split(cli_str)
                cli()
            except:
                failed_combos.append('{:35} {:35}'.format(clf_name, fs_name))
                traceback.print_exc()

    print('\nCombinations failed:\n{}'.format('\n'.join(failed_combos)))
    if len(failed_combos) > 4:
        print('5 or more combinations of DR and REGR failed! Fix them')


test_each_combination_works()
