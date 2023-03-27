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

from neuropredict.classify import cli
from neuropredict import config as cfg
from pyradigm import ClassificationDataset
from pyradigm.utils import (make_random_ClfDataset,
                            dataset_with_new_features_same_everything_else)

from neuropredict.tests._test_utils import (raise_if_mean_differs_from,
                                            remove_neuropredict_results)

feat_generator = np.random.randn

test_dir = Path(__file__).resolve().parent
out_dir = test_dir.joinpath('..', 'tests', 'scratch_classify')
if not out_dir.exists():
    out_dir.mkdir()

min_num_classes = 2
max_num_classes = 2
max_class_size = 200

max_dim = 100
min_rep_per_class = 10

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 10
num_procs = 1

red_dim = 'sqrt'
estimator = 'randomforestclassifier'  # 'svm' #
dr_method = 'isomap'  # 'selectkbest_f_classif' # 'variancethreshold'  #
dr_size = 'tenth'
gs_level = 'none'  # 'light'

random.seed(42)  # to save time for local tests

covar_list = ('age', 'gender', 'dummy')
covar_types = ('age', 'gender', 'float')
covar_arg = ' '.join(['age', 'gender'])
deconf_method = 'residualize'

out_path1 = out_dir / 'random_clf_ds1.pkl'
out_path2 = out_dir / 'random_clf_ds2.pkl'
if out_path1.exists() and out_path2.exists():
    ds_one = ClassificationDataset(dataset_path=out_path1)
    ds_two = ClassificationDataset(dataset_path=out_path2)
else:
    ds_one = make_random_ClfDataset(max_num_classes=max_num_classes,
                                    stratified=True,
                                    max_class_size=max_class_size,
                                    max_dim=max_dim,
                                    min_num_classes=min_num_classes,
                                    attr_names=covar_list,
                                    attr_types=covar_types)
    ds_one.save(out_path1)

    ds_two = dataset_with_new_features_same_everything_else(ds_one, max_dim)
    ds_two.save(out_path2)

A = 0
B = 1
C = 2
if ds_one.num_targets > 2:
    # sg_list =  '{},{} {},{} {}'.format(ds_one.target_set[A], ds_one.target_set[B],
    #                                    ds_one.target_set[A], ds_one.target_set[C],
    #                                    ','.join(ds_one.target_set))
    # sg_list = '{},{} {}'.format(ds_one.target_set[A], ds_one.target_set[B],
    #                             ','.join(ds_one.target_set))
    sg_list = ' {},{} '.format(ds_one.target_set[A], ds_one.target_set[B])
else:
    sg_list = ','.join(ds_one.target_set)

# sg_list = ' class-1,class-4 '

# choosing the class that exists in all subgroups
positive_class = ds_one.target_set[A]

# deciding on tolerances for chance accuracy
total_num_classes = ds_one.num_targets

eps_chance_acc_binary = 0.04
eps_chance_acc = max(0.02, 0.1 / total_num_classes)


def test_basic_run():
    sys.argv = shlex.split('np_classify -y {} {} -t {} -n {} -c {} -g {} -o {} '
                           '-e {} -dr {} -k {} --sub_groups {} -p {} -cl {} -cm {}'
                           ''.format(out_path1, out_path2, train_perc, num_rep_cv,
                                     num_procs, gs_level, out_dir,
                                     estimator, dr_method, dr_size,
                                     sg_list, positive_class,
                                     covar_arg, deconf_method))
    cli()


def test_chance_clf_binary_svm():
    sys.argv = shlex.split('neuropredict -y {} {} -t {} -n {} -c {} -g {} -o {} '
                           '-e {} -dr {}'
                           ''.format(out_path1, out_path2, train_perc,
                                     min_rep_per_class * ds_two.num_targets,
                                     num_procs, gs_level, out_dir, estimator,
                                     dr_method))
    result_paths = cli()
    import pickle
    for sg_id, res_path in result_paths.items():
        with open(res_path, 'rb') as res_fid:
            result = pickle.load(res_fid)

        perf = result['results']

        bal_acc_all_dsets = list(perf.metric_val['balanced_accuracy_score'].values())
        raise_if_mean_differs_from(np.column_stack(bal_acc_all_dsets),
                                   result['_target_sizes'],
                                   eps_chance_acc=eps_chance_acc_binary)


def test_each_combination_works():
    """Ensures each of combination of feature selection and classifier works."""

    nrep = 10
    nproc = 1
    gsl = 'none'  # to speed up the process
    failed_combos = list()
    for clf_name in cfg.classifier_choices:
        for fs_name in cfg.all_dim_red_methods:
            # ensure a fresh start
            remove_neuropredict_results(out_dir)
            try:
                cli_str = 'np_classify -y {} -t {} -n {} -c {} -o {} ' \
                          ' -e {} -dr {} -g {} ' \
                          ''.format(out_path1, train_perc, nrep, nproc, out_dir,
                                    clf_name, fs_name, gsl)
                sys.argv = shlex.split(cli_str)
                cli()
            except:
                failed_combos.append('{:35} {:35}'.format(clf_name, fs_name))
                traceback.print_exc()

    print('\nCombinations failed:\n{}'.format('\n'.join(failed_combos)))
    if len(failed_combos) > 4:
        print('\n  -----> 5 or more combinations of DR and CLF failed! Fix them')

