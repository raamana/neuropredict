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

feat_generator = np.random.randn

test_dir = dirname(os.path.realpath(__file__))
out_dir = realpath(pjoin(test_dir, '..', 'tests', 'scratch_classify'))
if not pexists(out_dir):
    os.makedirs(out_dir)

min_num_classes = 3
max_num_classes = 10
max_class_size = 200

max_dim = 100
min_rep_per_class = 20

min_num_modalities = 3
max_num_modalities = 10

train_perc = 0.5
num_rep_cv = 20

deconf_method = 'residualize'

red_dim = 'sqrt'
estimator = 'randomforestclassifier'
dr_method = 'variancethreshold'  # 'selectkbest_f_classif'
gs_level = 'none'  # 'light'

num_procs = 1

random.seed(42)  # to save time for local tests

covar_list = ('age', 'gender', 'dummy')
covar_types = ('age', 'gender', 'float')
covar_arg = ','.join(['age', ])

out_path = os.path.join(out_dir, 'random_clf_ds1.pkl')
if pexists(out_path):
    ds_one = ClassificationDataset(dataset_path=out_path)
else:
    ds_one = make_random_ClfDataset(max_num_classes=max_num_classes,
                                    stratified=True,
                                    max_class_size=max_class_size,
                                    max_dim=max_dim,
                                    min_num_classes=min_num_classes,
                                    attr_names=covar_list,
                                    attr_types=covar_types)
    ds_one.save(out_path)

A = 0
B = 1
C = 2
if ds_one.num_targets > 2:
    # sg_list =  '{},{} {},{} {}'.format(ds_one.target_set[A], ds_one.target_set[B],
    #                                    ds_one.target_set[A], ds_one.target_set[C],
    #                                    ','.join(ds_one.target_set))
    sg_list = '{},{} {}'.format(ds_one.target_set[A], ds_one.target_set[B],
                                ','.join(ds_one.target_set))
else:
    sg_list = ','.join(ds_one.target_set)

# choosing the class that exists in all subgroups
positive_class = ds_one.target_set[A]

out_path2 = os.path.join(out_dir, 'random_clf_ds2.pkl')
ds_two = dataset_with_new_features_same_everything_else(ds_one, max_dim)
ds_two.save(out_path2)

sys.argv = shlex.split('np_classify -y {} {} -t {} -n {} -c {} -g {} -o {} '
                       '-e {} -dr {} --sub_groups {} -p {} -cl {} -cm {}'
                       ''.format(out_path, out_path2, train_perc, num_rep_cv,
                                 num_procs, gs_level, out_dir,
                                 estimator, dr_method, sg_list, positive_class,
                                 covar_arg, deconf_method))
cli()
