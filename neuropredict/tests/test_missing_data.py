from os.path import abspath, dirname, exists as pexists, join as pjoin, realpath
import os
import shlex
import sys
import numpy as np
from neuropredict import cli as run_cli, config_neuropredict as cfg
from neuropredict.base import MissingDataException
from pytest import raises
from pyradigm import MLDataset

test_dir = dirname(os.path.realpath(__file__))
data_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

out_dir = pjoin(test_dir, 'missing')
os.makedirs(out_dir, exist_ok=True)

iris_path = pjoin(data_dir, 'pyradigm', 'iris.MLDataset.pkl')

ds_with_missing1 = MLDataset(filepath=iris_path)
ds_with_missing1['row141'] = [1, 2, np.NaN, 2]

ds_with_missing2 = MLDataset(filepath=iris_path)
ds_with_missing2['row043'] = [1, np.NaN, 2, 5]

ds_with_missing1_path = pjoin(out_dir, 'trial1.Iris.MLDataset.pkl')
ds_with_missing2_path = pjoin(out_dir, 'trial2.Iris.MLDataset.pkl')

ds_with_missing1.save(ds_with_missing1_path)
ds_with_missing2.save(ds_with_missing2_path)

ds_list = [ds_with_missing1_path, ds_with_missing2_path]


def test_raises_missing_data_error():
    """When no imputation is chosen, checking that an error is raised."""

    for ds_path in ds_list:
        # default is to raise, even if not specified
        with raises((MissingDataException,)):
            sys.argv = shlex.split('neuropredict -e svm -y {} -o {}'.format(ds_path, out_dir))
            run_cli()

        # checking explicit raise
        with raises((MissingDataException,)):
            sys.argv = shlex.split('neuropredict -y {} -o {} --impute_strategy raise '
                                   ''.format(ds_path, out_dir))
            run_cli()


def test_imputation_works_when_chosen():
    for strategy in cfg.avail_imputation_strategies:
        try:
            sys.argv = shlex.split('neuropredict -n 10 -t 0.5 --impute_strategy {} '
                                   '-y {} -o {}'.format(strategy,
                                                        ds_with_missing1_path, out_dir))
            run_cli()
        except:
            raise RuntimeError('Imputation with {} strategy failed!'.format(strategy))

