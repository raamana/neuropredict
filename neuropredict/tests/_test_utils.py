from pathlib import Path

import numpy as np
from neuropredict import config as cfg
from neuropredict.utils import chance_accuracy


def raise_if_mean_differs_from(accuracy_balanced,
                               class_sizes,
                               reference_level=None,
                               eps_chance_acc=None,
                               method_descr=''):
    """
    Check if the performance is close to chance.

    Generic method that works for multi-class too!"""

    if eps_chance_acc is None:
        total_num_classes = len(class_sizes)
        eps_chance_acc = max(0.02, 0.1 / total_num_classes)

    if reference_level is None:
        reference_level = chance_accuracy(class_sizes)
    elif not 0.0 < reference_level <= 1.0:
        raise ValueError('invalid reference_level: must be in (0, 1]')

    # chance calculation expects "average", not median
    mean_bal_acc = np.mean(accuracy_balanced, axis=0)
    for ma in mean_bal_acc:
        print('for {},\n reference level accuracy expected: {} '
              '-- Estimated via CV:  {}'.format(method_descr, reference_level, ma))
        abs_diff = abs(ma - reference_level)
        if abs_diff > eps_chance_acc:
            raise ValueError('they substantially differ by {:.4f} that is '
                             'more than {:.4f}!'.format(abs_diff, eps_chance_acc))


def remove_neuropredict_results(out_dir):
    """Removes existing results to ensure subsequent runs result in fresh results"""

    for rf in Path(out_dir).rglob(cfg.results_file_name):
        try:
            rf.unlink()
        except:
            print('Unable to delete {}'.format(rf))
