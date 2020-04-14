import pickle
import numpy as np
from pathlib import Path
from os import makedirs
from neuropredict import config as cfg
from neuropredict.results import ClassifyCVResults, RegressCVResults
from neuropredict.visualize import compare_distributions
from neuropredict.classify import ClassificationWorkflow as ClfWorkflow

this_dir = Path(__file__).parent.resolve()

def load_results(results_path):

    try:
        with open(results_path, 'rb') as res_fid:
            full_results = pickle.load(res_fid)
    except:
        raise IOError()

    return full_results


def _test_shape_finiteness(results, num_rep_cv, num_datasets):
    """"""

    def shape_match(arr):
        return (arr.shape[0] == num_rep_cv) and \
               (arr.shape[1] == num_datasets) and \
               (len(ds_ids) == num_datasets)

    for metric in list(results.metric_set):
        arr, ds_ids = results.to_array(metric)

        if not shape_match(arr):
            raise ValueError('Invalid shape for array form of {} : {}; It must be {}'
                             ''.format(metric, arr.shape,
                                       (num_rep_cv, num_datasets)))

        if not np.isfinite(arr).all():
            raise ValueError('some estimates of {} are infinite/NaN!'.format(metric))


def test_classify():

    res_dir = this_dir / 'scratch_classify'
    # ** in glob --> recursive subdir search
    result_paths = list(res_dir.glob('**/{}'.format(cfg.results_file_name)))

    for rr, rpath in enumerate(result_paths):
        full_results = load_results(rpath)
        print('testing results from {}'
              ''.format(rpath))
              # ''.format(full_results['user_options']['sub_groups'][rr], rpath))

        num_datasets = len(full_results['user_options']['user_feature_paths'])
        num_targets = len(full_results['_target_set'])

        clf_res = ClassifyCVResults(path=rpath)
        _test_shape_finiteness(clf_res, full_results['num_rep_cv'], num_datasets)

        for mt_full, mt_short in zip(('balanced_accuracy_score', 'area_under_roc'),
                                     ('balanced accuracy', 'AUC')):
            if num_targets > 2 and 'AUC' in mt_short:
                continue
            if mt_full not in clf_res.metric_set:
                print('metric {} does not seem to exist in {}'.format(mt_full, rpath))
                continue

            arr, ds_ids = clf_res.to_array(mt_full)
            if arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError('some estimates of {} are out of bounds'
                                 ' i.e. <0.0 or >1.0'.format(mt_short))


def test_regress():

    result_path = this_dir / 'scratch_regress' / cfg.results_file_name
    full_results = load_results(result_path)

    num_datasets = len(full_results['user_options']['user_feature_paths'])

    regr_results = RegressCVResults(path=result_path)
    _test_shape_finiteness(regr_results, full_results['num_rep_cv'], num_datasets)

    print()


test_classify()
