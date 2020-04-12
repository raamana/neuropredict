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

from neuropredict.classify import cli
from pyradigm import ClassificationDataset
from pyradigm.utils import (make_random_ClfDataset,
                            dataset_with_new_features_same_everything_else)

from neuropredict.tests._test_utils import raise_if_mean_differs_from

feat_generator = np.random.randn

test_dir = dirname(os.path.realpath(__file__))
out_dir = realpath(pjoin(test_dir, '..', 'tests', 'scratch_classify'))
if not pexists(out_dir):
    os.makedirs(out_dir)

min_num_classes = 3
max_num_classes = 5
max_class_size = 200

max_dim = 100
min_rep_per_class = 10

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 40
num_procs = 2

red_dim = 'sqrt'
estimator =  'randomforestclassifier' # 'svm' #
dr_method = 'isomap' # 'selectkbest_f_classif' # 'variancethreshold'  #
dr_size = 'tenth'
gs_level = 'none'  # 'light'

random.seed(42)  # to save time for local tests

covar_list = ('age', 'gender', 'dummy')
covar_types = ('age', 'gender', 'float')
covar_arg = ' '.join(['age', 'gender'])
deconf_method = 'residualize'


out_path1 = os.path.join(out_dir, 'random_clf_ds1.pkl')
out_path2 = os.path.join(out_dir, 'random_clf_ds2.pkl')
if pexists(out_path1) and pexists(out_path2):
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

eps_chance_acc_binary =0.04
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

test_basic_run()
