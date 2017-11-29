
import numpy as np
import os
import sys
from sys import version_info
import shlex
from os.path import join as pjoin, exists as pexists, realpath, dirname, abspath

sys.dont_write_bytecode = True

from pyradigm import MLDataset
import pytest
from pytest import raises, warns

if __name__ == '__main__' and __package__ is None:
    parent_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(parent_dir)

if version_info.major > 2:
    import neuropredict
    from neuropredict import rhst, cli, config_neuropredict as cfg
    from neuropredict.utils import chance_accuracy
else:
    raise NotImplementedError('neuropredict supports only Python 3+.')

feat_generator = np.random.randn

out_dir = os.path.abspath('../tests/scratch')
if not pexists(out_dir):
    os.makedirs(out_dir)

meta_file = os.path.join(out_dir,'meta.csv')

meta = list()

def make_random_MLdataset(max_num_classes = 20,
                          max_class_size = 50,
                          max_dim = 100,
                          stratified = True):
    "Generates a random MLDataset for use in testing."

    smallest = 10
    max_class_size = max(smallest, max_class_size)
    largest = max(50, max_class_size)
    largest = max(smallest+3,largest)

    num_classes = np.random.randint(2, max_num_classes, 1)
    if type(num_classes) == np.ndarray:
        num_classes = num_classes[0]
    if not stratified:
        class_sizes = np.random.random_integers(smallest, largest,
                                                size=[num_classes, 1])
    else:
        class_sizes = np.repeat(np.random.randint(smallest, largest),
                                                  num_classes)

    num_features = np.random.randint(min(3, max_dim), max(3, max_dim), 1)[0]
    feat_names = [ str(x) for x in range(num_features)]

    class_ids = list()
    labels = list()
    for cl in range(num_classes):
        class_ids.append('class-{}'.format(cl))
        labels.append(int(cl))

    ds = MLDataset()
    for cc, class_ in enumerate(class_ids):
        subids = [ 'sub{:03}-class{:03}'.format(ix,cc) for ix in range(class_sizes[cc]) ]
        for sid in subids:
            ds.add_sample(sid, feat_generator(num_features), int(cc), class_, feat_names)

    return ds


max_num_classes = 10
max_class_size = 40
max_dim = 100
num_repetitions = 20
min_rep_per_class = 10

train_perc = 0.5
red_dim = 'sqrt'
classifier = 'svm' # 'extratreesclassifier'
fs_method = 'variancethreshold' # 'selectkbest_f_classif'
gs_level = 'none'

num_procs = 1


# using a really small sample size for faster testing.
rand_ds = make_random_MLdataset(max_num_classes=max_num_classes, stratified=True,
    max_class_size = max_class_size, max_dim = max_dim)

out_path_multiclass = os.path.join(out_dir, 'multiclass_random_features.pkl')
rand_ds.save(out_path_multiclass)

out_path = os.path.join(out_dir, 'two_classes_random_features.pkl')
rand_two_class = rand_ds.get_class(rand_ds.class_set[0:3])
rand_two_class.description = 'random_1'
rand_two_class.save(out_path)

# making another copy to a different path with different description
rand_two_class.description = 'random_2'
out_path2 = os.path.join(out_dir, 'two_classes_random_features_another.pkl')
rand_two_class.save(out_path2)

ds_path_list = os.path.join(out_dir, 'same_data_two_classes_list_datasets.txt')
with open(ds_path_list, 'w') as lf:
    lf.writelines('\n'.join([out_path, out_path2]))

method_names = ['random1', 'another']

# deciding on tolerances for chance accuracy
total_num_classes = rand_ds.num_classes

eps_chance_acc_binary =0.05
eps_chance_acc = max(0.02, 0.1 / total_num_classes)


def raise_if_mean_differs_from_chance(accuracy_balanced, class_sizes,
                                      eps_chance_acc=None):
    "Check if the performance is close to chance. Generic method that works for multi-class too!"

    if eps_chance_acc is None:
        total_num_classes = len(class_sizes)
        eps_chance_acc = max(0.02, 0.1 / total_num_classes)

    chance_acc = chance_accuracy(class_sizes)
    # chance calculation expects "average", not median
    mean_bal_acc = np.mean(accuracy_balanced, axis=0)
    for ma  in mean_bal_acc:
        print('Chance accuracy expected: {} -- Estimated via CV:  {}'.format(chance_acc, ma))
        if abs(ma - chance_acc) > eps_chance_acc:
            raise ValueError('they substantially differ by more than {:.4f}!'.format(eps_chance_acc))


def test_chance_clf_binary_svm():

    global ds_path_list, method_names, out_dir, num_repetitions, gs_level, train_perc, num_procs

    sys.argv = shlex.split('neuropredict -y {} {} -t {} -n {} -c {} -g {} -o {} -e {} -fs {}'.format(out_path, out_path2,
                                train_perc, min_rep_per_class*rand_two_class.num_classes, num_procs, gs_level,
                                out_dir, classifier, fs_method))
    cli()

    cv_results = rhst.load_results_from_folder(out_dir)
    for sg, result in cv_results.items():
        raise_if_mean_differs_from_chance(result['accuracy_balanced'],
                                          result['class_sizes'],
                                          eps_chance_acc_binary)


def test_chance_multiclass():

    global ds_path_list, method_names, out_dir, num_repetitions, gs_level, train_perc, num_procs

    clf = 'randomforestclassifier'
    fs_method = 'variancethreshold'
    nrep = total_num_classes*min_rep_per_class
    gsl = 'none'  # to speed up the process
    sys.argv = shlex.split('neuropredict -y {} -t {} -n {} -c {} -g {} -o {} -e {} -fs {}'.format(out_path_multiclass,
                                train_perc, nrep, num_procs, gsl, out_dir, clf, fs_method))
    cli()

    cv_results = rhst.load_results_from_folder(out_dir)
    for sg, result in cv_results.items():
        raise_if_mean_differs_from_chance(result['accuracy_balanced'], result['class_sizes'], eps_chance_acc)


def test_each_combination_works():
    "Ensures each of combination of feature selection and classifier works."

    nrep = 10
    gsl = 'none' # to speed up the process
    for clf_name in cfg.classifier_choices:
        for fs_name in cfg.feature_selection_choices:
            try:
                cli_str = 'neuropredict -y {} -t {} -n {} -c {} -o {}  -e {} -fs {} -g {} '.format(out_path,
                            train_perc, nrep, num_procs, out_dir, clf_name, fs_name, gsl)
                sys.argv = shlex.split(cli_str)
                cli()
            except:
                print(' ---> combination failed: {} {}'.format(clf_name, fs_name))
                raise


def test_versioning():
    " ensures the CLI works. "

    with raises(SystemExit):
        sys.argv = shlex.split('neuropredict -v')
        cli()

def test_vis():
    " ensures the CLI works. "

    res_path = pjoin(out_dir, 'rhst_results.pkl')
    if pexists(res_path):
        with raises(SystemExit):
            sys.argv = shlex.split('neuropredict --make_vis {}'.format(out_dir))
            cli()
            expected_results = ['balanced_accuracy.pdf', 'compare_misclf_rates.pdf', 'feature_importance.pdf']
            for rpath in expected_results:
                if not pexists(rpath):
                    raise ValueError('an expected result {} not produced'.format(rpath))
    else:
        print('previously computed results not found in \n {}'.format(out_dir))

def test_arff():

    arff_path = realpath(pjoin(dirname(dirname(dirname(__file__))), # 3 levels up
                               'example_datasets', 'arff', 'iris.arff'))
    sys.argv = shlex.split('neuropredict -a {} -t {} -n {} -c {} -g {} -o {} -e {} -fs {}'.format(arff_path,
                    train_perc, num_repetitions, num_procs, gs_level, out_dir, classifier, fs_method))
    cli()


# res_path = pjoin(out_dir, 'rhst_results.pkl')
# run_workflow.make_visualizations(res_path, out_dir)
# test_chance_clf_default()
# test_chance_clf_binary_extratrees()
test_chance_clf_binary_svm()
# test_chance_multiclass()
# test_versioning()
# test_vis()

# etc_debug()

# test_arff()
