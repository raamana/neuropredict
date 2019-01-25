from os.path import abspath, dirname, exists as pexists, join as pjoin, realpath
import os
import shlex
import sys
from neuropredict import cli as run_cli, config_neuropredict as cfg
from neuropredict.run_workflow import MissingDataException
from pytest import raises

# in_dir = '/Volumes/data/work/rotman/CANBIND/data/Tier1_v1/processing'
in_dir = '/home/praamana/CANBIND'
out_dir = pjoin(in_dir, 'missing')
os.makedirs(out_dir, exist_ok=True)

ds_with_missing1 = pjoin(in_dir, 'CANBIND_Tier1_Clinical_features.KeepNaN.MLDataset.pkl')
ds_with_missing2 = pjoin(in_dir, 'CANBIND_Tier1_Molecular_features.KeepNaN.MLDataset.pkl')

ds_list = [ds_with_missing1, ds_with_missing2]


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
                                   '-y {} -o {}'.format(strategy, ds_with_missing1, out_dir))
            run_cli()
        except:
            raise RuntimeError('Imputation with {} strategy failed!'.format(strategy))

