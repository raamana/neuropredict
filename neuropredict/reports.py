import os
import pickle
import shutil
import traceback
from collections import Counter
from os.path import exists as pexists, join as pjoin

import numpy as np
from neuropredict import config as cfg, visualize
from neuropredict.utils import load_options, sub_group_identifier


def export_results(dict_to_save, out_dir, options_path):
    """
    Exports the results to simpler CSV format for use in other packages!

    Parameters
    ----------
    dict_to_save : dict
        Containing all the relevant results

    out_dir : str
        Path to save the results to.

    Returns
    -------
    None

    """

    confusion_matrix        = dict_to_save['confusion_matrix']
    accuracy_balanced       = dict_to_save['accuracy_balanced']
    method_names            = dict_to_save['method_names']
    feature_importances_rf  = dict_to_save['feature_importances_rf']
    feature_names           = dict_to_save['feature_names']
    num_times_misclfd       = dict_to_save['num_times_misclfd']
    num_times_tested        = dict_to_save['num_times_tested']

    num_rep_cv   = confusion_matrix.shape[0]
    num_datasets = confusion_matrix.shape[3]
    num_classes  = confusion_matrix.shape[2]

    # separating CSVs from the PDFs
    exp_dir = pjoin(out_dir, cfg.EXPORT_DIR_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # TODO think about how to export predictive probability per class per CV rep
    # pred_prob_per_class

    user_options = load_options(out_dir, options_path)
    print_aligned_msg = lambda msg1, msg2: print('Exporting {msg1:<40} .. {msg2}'
                                                 ''.format(msg1=msg1, msg2=msg2))

    print('')
    try:
        # accuracy
        balacc_path = pjoin(exp_dir, 'balanced_accuracy.csv')
        np.savetxt(balacc_path, accuracy_balanced,
                   delimiter=cfg.DELIMITER,
                   fmt=cfg.EXPORT_FORMAT,
                   header=','.join(method_names))
        print_aligned_msg('accuracy distribution', 'Done.')

        # conf mat
        for mm in range(num_datasets):
            confmat_path = pjoin(exp_dir, 'confusion_matrix_{}.csv'
                                          ''.format(method_names[mm]))
            reshaped_matrix = np.reshape(confusion_matrix[:, :, :, mm],
                                         [num_rep_cv, num_classes * num_classes])
            np.savetxt(confmat_path, reshaped_matrix,
                       delimiter=cfg.DELIMITER, fmt=cfg.EXPORT_FORMAT,
                       comments='shape of confusion matrix: '
                                'num_repetitions x num_classes^2')
        print_aligned_msg('confusion matrices', 'Done.')

        # misclassfiication rates
        avg_cfmat, misclf_rate = visualize.compute_pairwise_misclf(confusion_matrix)
        num_datasets = misclf_rate.shape[0]
        for mm in range(num_datasets):
            cmp_misclf_path = pjoin(exp_dir, 'average_misclassification_rates_{}.csv'
                                             ''.format(method_names[mm]))
            np.savetxt(cmp_misclf_path,
                       misclf_rate[mm, :],
                       fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER)
        print_aligned_msg('misclassfiication rates', 'Done.')

        # feature importance
        if user_options['pred_model_name'].lower() in cfg.estimators_with_feat_imp:
            for mm in range(num_datasets):
                featimp_path = pjoin(exp_dir, 'feature_importance_{}.csv'
                                              ''.format(method_names[mm]))
                np.savetxt(featimp_path,
                           feature_importances_rf[mm],
                           fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER,
                           header=','.join(feature_names[mm]))
            print_aligned_msg('feature importance values', 'Done.')

        else:
            print_aligned_msg('feature importance values', 'Skipped.')
            print('\tCurrent predictive model does not provide them.')

        # subject-wise misclf frequencies
        perc_misclsfd, _, _, _ = visualize.compute_perc_misclf_per_sample(
                num_times_misclfd, num_times_tested)
        for mm in range(num_datasets):
            subwise_misclf_path = pjoin(exp_dir, 'subject_misclf_freq_{}.csv'
                                                 ''.format(method_names[mm]))
            # TODO there must be a more elegant way to write dict to CSV
            with open(subwise_misclf_path, 'w') as smf:
                for sid, val in perc_misclsfd[mm].items():
                    smf.write('{}{}{}\n'.format(sid, cfg.DELIMITER, val))
        print_aligned_msg('subject-wise misclf frequencies', 'Done.')

    except:
        traceback.print_exc()
        raise IOError('Unable to export the results to CSV files.')

    return


def report_best_params(best_params, method_names, out_dir):
    """Outputs most frequently selected parameter values to stdout and disk."""

    # best_params : list of num_reps elements, each a list of num_datasets dicts
    num_reps = len(best_params)
    param_names = list(best_params[0][0].keys())

    # building a list of best values (from each rep) for each parameter
    param_values = {label: dict() for label in method_names}
    for rep in range(num_reps):
        for ds, label in enumerate(method_names):
            param_values[label] = {param: list() for param in param_names}
            for param in best_params[rep][ds]:
                param_values[label][param].append(best_params[rep][ds][param])

    # finding frequent values and printing them
    print('\nThe most frequently selected parameter values are: ')
    most_freq_values = {label: dict() for label in method_names}
    maxwidth = 2 + max([len(p) for p in param_names])
    for ds, label in enumerate(method_names):
        print('  For feature set {} : {}'.format(ds, label))
        for param in param_names:
            pcounter = Counter(param_values[label][param])
            # most_common returns a list of tuples, with value and its frequency
            most_freq_values[label][param] = pcounter.most_common(1)[0][0]
            print('    {:{mw}} : {}'
                  ''.format(param, most_freq_values[label][param], mw=maxwidth))
        print('')

    # saving them
    try:
        out_results_path = pjoin(out_dir, cfg.file_name_best_param_values)
        with open(out_results_path, 'wb') as resfid:
            pickle.dump(most_freq_values, resfid)
    except:
        raise IOError('Error saving the results to disk!')

    return


def export_results_from_disk(results_file_path, out_dir, options_path):
    """
    Exports the results to simpler CSV format for use in other packages!

    Parameters
    ----------
    results_file_path : str
        Path to a pickle file containing all the relevant results

    out_dir : str
        Path to save the results to

    Returns
    -------
    None

    """

    dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, best_params, \
    feature_importances_rf, feature_names, num_times_misclfd, num_times_tested, \
    confusion_matrix, class_order, class_sizes, accuracy_balanced, auc_weighted, \
    positive_class = classifier_name, feat_select_method = load_results(
        results_file_path)

    locals_var_dict = locals()
    dict_to_save = {var: locals_var_dict[var]
                    for var in cfg.rhst_data_variables_to_persist}
    export_results(dict_to_save, out_dir, options_path)

    return


def save_results(out_dir, dict_of_objects_to_save):
    "Serializes the results to disk."

    # LATER choose a more universal serialization method
    #   (that could be loaded from a web app)
    try:
        out_results_path = pjoin(out_dir, cfg.results_file_name)
        with open(out_results_path, 'wb') as resfid:
            pickle.dump(dict_of_objects_to_save, resfid)
    except:
        raise IOError('Error saving the results to disk!')
    else:
        # deleting temp results only when saving full results is successful
        cleanup(out_dir)

    return out_results_path


def load_results_from_folder(results_folder):
    """

    Given a base output folder, possibly containing results for multiple sub-groups,
        returns a dictionary of results, keyed in by sub group identifier.

    """

    results = dict()
    options = load_options(results_folder)
    for ix, sg in enumerate(options['sub_groups']):
        sg_id = sub_group_identifier(sg, ix)
        results_file_path = pjoin(results_folder, sg_id, cfg.results_file_name)
        if not pexists(results_file_path) or os.path.getsize(results_file_path) <= 0:
            raise IOError('Results file for sub group {} does not exist'
                          ' or is empty!'.format(sg_id))
        results[sg_id] = load_results_dict(results_file_path)

    return results


def load_results_dict(results_file_path):
    "Loads the results serialized by RHsT."
    # TODO need to standardize what needs to saved/read back

    if not pexists(results_file_path) or os.path.getsize(results_file_path) <= 0:
        raise IOError("Results file to be loaded doesn't exist, or empty!")

    try:
        with open(results_file_path, 'rb') as rf:
            results_dict = pickle.load(rf)
    except:
        raise IOError('Error loading the saved results from \n{}'
                      ''.format(results_file_path))

    return results_dict


def load_results(results_file_path):
    "Loads the results serialized by RHsT."
    # TODO need to standardize what needs to saved/read back

    if not pexists(results_file_path):
        raise IOError("Results file to be loaded doesn't exist!")

    try:
        with open(results_file_path, 'rb') as rf:
            results_dict = pickle.load(rf)
            # # below is possible, but not explicit and a bad practice
            # # importing the keys and their values into the workspace
            # locals().update(results_dict)

            dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
            pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
            best_params, feature_importances_rf, \
            feature_names, num_times_misclfd, num_times_tested, \
            confusion_matrix, class_set, target_sizes, accuracy_balanced, \
            auc_weighted, positive_class, classifier_name, feat_select_method = \
                [results_dict.get(var_name)
                 for var_name in cfg.rhst_data_variables_to_persist]

    except:
        raise IOError('Error loading the saved results from \n{}'
                      ''.format(results_file_path))

    # TODO need a consolidated way to identify the variables saved their order
    return dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
           pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
           best_params, feature_importances_rf, feature_names, \
           num_times_misclfd, num_times_tested, \
           confusion_matrix, class_set, target_sizes, \
           accuracy_balanced, auc_weighted, positive_class, \
           classifier_name, feat_select_method


def get_temp_dir(out_results_dir):
    "Scratch directory to save temporary results to"

    tmp_dir = pjoin(out_results_dir, cfg.temp_results_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    return tmp_dir


def cleanup(out_dir):
    "Helper to perform cleanup"

    tmp_dir = get_temp_dir(out_dir)
    try:
        shutil.rmtree(tmp_dir)
    except:
        traceback.print_exc()
        print('Unable to delete temporary folder at:\n\t{}\n'
              'Remove it manually if you would like to save space.'.format(tmp_dir))

    return
