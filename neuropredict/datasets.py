"""Module to load, check and combine datasets."""

import numpy as np
from warnings import warn
from pyradigm import MultiDatasetClassify, MultiDatasetRegress
from neuropredict import config as cfg
from neuropredict.base import MissingDataException


def load_datasets(path_list, task_type='classify', name='Dataset', subgroup=None):
    """Method to manage multiple input datasets."""

    task_type = task_type.lower()

    if task_type in ('classify',):
        multi_ds = MultiDatasetClassify(dataset_spec=path_list, name=name,
                                        subgroup=subgroup)
    elif task_type in ('regress',):
        if subgroup is not None:
            warn('Invalid specification of subgroup for regression datasets/tasks.'
                 ' Ignoring it.')
        multi_ds = MultiDatasetRegress(dataset_spec=path_list, name=name)
    else:
        raise ValueError('Invalid task type. Choose either classify or regress')

    return multi_ds


def detect_missing_data(multi_ds,
                        user_impute_strategy=cfg.default_imputation_strategy):
    """Detect and impute missing data"""

    missing_data_flag = dict()

    for ds_id, data_mat in multi_ds:

        is_nan = np.isnan(data_mat)
        if is_nan.any():
            data_missing_here = True
            num_sub_with_md = np.sum(is_nan.sum(axis=1) > 0)
            num_var_with_md = np.sum(is_nan.sum(axis=0) > 0)
            if user_impute_strategy == 'raise':
                raise MissingDataException(
                    '{}/{} subjects with missing data found in {}/{} features\n'
                    '\tFill them and rerun, '
                    'or choose one of the available imputation strategies: {}'
                    ''.format(num_sub_with_md, data_mat.shape[0],
                              num_var_with_md, data_mat.shape[1],
                              cfg.avail_imputation_strategies))
        else:
            data_missing_here = False

        multi_ds.set_attr(ds_id, cfg.missing_data_flag_name, data_missing_here)
        missing_data_flag[ds_id] = data_missing_here

    # finalizing the imputation strategy
    if any(missing_data_flag.values()):
        print('\nOne or more of the input datasets have missing data!')
        if user_impute_strategy == 'raise':
            raise MissingDataException('Fill them and rerun, '
                                       'or choose one of the available '
                                       'imputation strategies: {}'
                                       ''.format(cfg.avail_imputation_strategies))
        else:
            impute_strategy = user_impute_strategy
            print('The imputation strategy chosen is: {}'.format(impute_strategy))
    else:
        # disabling the imputation altogether if there is no missing data
        impute_strategy = None
        if user_impute_strategy in ('raise', None):
            print('\nIgnoring imputation strategy chosen,'
                  ' as no missing data were found!\n')

    return impute_strategy


