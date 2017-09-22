from __future__ import print_function

__all__ = ['fit', 'run_cli', 'get_parser']

import argparse
import os
import sys
import re
import traceback
import warnings
from collections import Counter
from time import localtime, strftime
import matplotlib.pyplot as plt
from sys import version_info
from os.path import join as pjoin, exists as pexists, abspath, realpath, dirname, basename

import numpy as np
from pyradigm import MLDataset

# the order of import is very important to avoid circular imports
if version_info.major == 2 and version_info.minor == 7:
    import config_neuropredict as cfg
    import rhst, visualize
    from freesurfer import aseg_stats_subcortical, aseg_stats_whole_brain
elif version_info.major > 2:
    # the order of import is very important to avoid circular imports
    from neuropredict import config_neuropredict as cfg
    from neuropredict import rhst, visualize, freesurfer, model_comparison
    from neuropredict.freesurfer import aseg_stats_subcortical, aseg_stats_whole_brain
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')


def make_time_stamp():
    "Returns a timestamp string."

    # just by the hour
    return strftime('%Y%m%d-T%H', localtime())


def not_unspecified(var):
    """ Checks for null values of a give variable! """

    return var not in ['None', None, '']


def get_parser():
    "Parser to specify arguments and their defaults."

    parser = argparse.ArgumentParser(prog="neuropredict")

    help_text_fs_dir = """Absolute path to ``SUBJECTS_DIR`` containing the finished runs of Freesurfer parcellation (each subject named after its ID in the metadata file). E.g. ``--fs_subject_dir /project/freesurfer_v5.3`` """

    help_text_user_defined_folder = """List of absolute paths to user's own features. 
    
Format: Each of these folders contains a separate folder for each subject (named after its ID in the metadata file) containing a file called features.txt with one number per line. All the subjects (in a given folder) must have the number of features (#lines in file). Different parent folders (describing one feature set) can have different number of features for each subject, but they must all have the same number of subjects (folders) within them. 
    
Names of each folder is used to annotate the results in visualizations. Hence name them uniquely and meaningfully, keeping in mind these figures will be included in your papers. For example,  

.. parsed-literal::
    
    --user_feature_paths /project/fmri/ /project/dti/ /project/t1_volumes/ 

Only one of ``--pyradigm_paths``, ``user_feature_paths`` and ``daa_matrix_path`` options can be specified. """

    help_text_pyradigm_paths = """ Path(s) to pyradigm datasets. Each path is self-contained dataset identifying each sample, its class and features. """

    help_text_data_matrix = """List of absolute paths to text files containing one matrix of size N x p  (num_samples x num_features).  Each row in the data matrix file must represent data corresponding  to sample in the same row of the meta data file (meta data file and data matrix must be in row-wise correspondence). Name of this file will be used to annotate the results and visualizations.

E.g. ``--data_matrix_paths /project/fmri.csv /project/dti.csv /project/t1_volumes.csv ``
    
Only one of ``--pyradigm_paths``, ``user_feature_paths`` and ``daa_matrix_path`` options can be specified.  
File format could be 
 - a simple comma-separated text file (with extension .csv or .txt): which can easily be read back with numpy.loadtxt(filepath, delimiter=',') or 
 - a numpy array saved to disk (with extension .npy or .numpy) that can read in with numpy.load(filepath). 
 
 One could use ``numpy.savetxt(data_array, delimiter=',')`` or ``numpy.save(data_array)`` to save features. File format is inferred from its extension."""

    help_text_positive_class = "Name of the positive class (Alzheimers, MCI or Parkinsons etc) to be used in calculation of area under the ROC curve. Applicable only for binary classification experiments. Default: class appearning second in order specified in metadata file."
    help_text_train_perc = "Percentage of the smallest class to be reserved for training. Must be in the interval [0.01 0.99].If sample size is sufficiently big, we recommend 0.5.If sample size is small, or class imbalance is high, choose 0.8."

    help_text_num_rep_cv = """Number of repetitions of the repeated-holdout cross-validation. The larger the number, the better the estimates will be."""

    help_text_sub_groups = """This option allows the user to study different combinations of classes in a multi-class (N>2) dataset. For example, in a dataset with 3 classes CN, FTD and AD, two studies of pair-wise combinations can be studied separately with the following flag ``--sub_groups CN,FTD CN,AD``. This allows the user to focus on few interesting subgroups depending on their dataset/goal. 
    
Format: Different subgroups must be separated by space, and each sub-group must be a comma-separated list of class names defined in the meta data file. Hence it is strongly recommended to use class names without any spaces, commas, hyphens and special characters, and ideally just alphanumeric characters separated by underscores. Any number of subgroups can be specified, but each subgroup must have atleast two distinct classes. 

Default: ``'all'``, leading to inclusion of all available classes in a all-vs-all multi-class setting."""

    help_text_metadata_file = """Abs path to file containing metadata for subjects to be included for analysis. At the minimum, each subject should have an id per row followed by the class it belongs to.

E.g.
.. parsed-literal::

    sub001,control
    sub002,control
    sub003,disease
    sub004,disease
    
    """

    help_text_feature_selection = """Number of features to select as part of feature selection. Options:

 - 'tenth'
 - 'sqrt'
 - 'log2'
 - 'all'

Default: \'tenth\' of the number of samples in the training set. For example, if your dataset has 90 samples, you chose 50 percent for training (default),  then Y will have 90*.5=45 samples in training set, leading to 5 features to be selected for taining. If you choose a fixed integer, ensure all the feature sets under evaluation have atleast that many features."""

    help_text_atlas = "Name of the atlas to use for visualization. Default: fsaverage, if available."

    parser.add_argument("-m", "--meta_file", action="store", dest="meta_file",
                        default=None, required=True,
                        help=help_text_metadata_file)

    parser.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=True,
                        help="Output folder to store gathered features & results.")

    parser.add_argument("-f", "--fs_subject_dir", action="store", dest="fs_subject_dir",
                        default=None,
                        help=help_text_fs_dir)

    user_defined = parser.add_mutually_exclusive_group()

    user_defined.add_argument("-y", "--pyradigm_paths", action="store", dest="pyradigm_paths",
                              nargs='+',  # to allow for multiple features
                              default=None,
                              help=help_text_pyradigm_paths)

    user_defined.add_argument("-u", "--user_feature_paths", action="store", dest="user_feature_paths",
                              nargs='+',  # to allow for multiple features
                              default=None,
                              help=help_text_user_defined_folder)

    user_defined.add_argument("-d", "--data_matrix_paths", action="store", dest="data_matrix_paths",
                              nargs='+',
                              default=None,
                              help=help_text_data_matrix)

    parser.add_argument("-p", "--positive_class", action="store", dest="positive_class",
                        default=None,
                        help=help_text_positive_class)

    parser.add_argument("-t", "--train_perc", action="store", dest="train_perc",
                        default=0.5,
                        help=help_text_train_perc)

    parser.add_argument("-n", "--num_rep_cv", action="store", dest="num_rep_cv",
                        default=200,
                        help=help_text_num_rep_cv)

    parser.add_argument("-k", "--num_features_to_select", dest="num_features_to_select",
                        action="store", default=cfg.default_num_features_to_select,
                        help=help_text_feature_selection)

    parser.add_argument("-a", "--atlas", action="store", dest="atlasid",
                        default="fsaverage",
                        help=help_text_atlas)

    parser.add_argument("-s", "--sub_groups", action="store", dest="sub_groups",
                        nargs="*",
                        default="all",
                        help=help_text_sub_groups)

    return parser


def organize_inputs(user_args):
    """
    Validates the input features specified and returns organized list of paths and readers.

    Parameters
    ----------
    user_args : ArgParse object
        Various options specified by the user.

    Returns
    -------
    user_feature_paths : list
        List of paths to specified input features
    user_feature_type : str
        String identifying the type of user-defined input
    fs_subject_dir : str
        Path to freesurfer subject directory, if supplied.

    """

    atleast_one_feature_specified = False
    if not_unspecified(user_args.fs_subject_dir):
        fs_subject_dir = abspath(user_args.fs_subject_dir)
        if not pexists(fs_subject_dir):
            raise IOError("Given Freesurfer directory doesn't exist.")
        atleast_one_feature_specified = True
    else:
        fs_subject_dir = None

    if not_unspecified(user_args.user_feature_paths):
        user_feature_paths = list(map(abspath, user_args.user_feature_paths))
        for udir in user_feature_paths:
            if not pexists(udir):
                raise IOError("One of the user directories for features doesn't exist:\n {}".format(udir))

        atleast_one_feature_specified = True
        user_feature_type = 'dir_of_dirs'

    elif not_unspecified(user_args.data_matrix_paths):
        user_feature_paths = list(map(abspath, user_args.data_matrix_paths))
        for dm in user_feature_paths:
            if not pexists(dm):
                raise IOError("One of the data matrices specified does not exist:\n {}".format(dm))

        atleast_one_feature_specified = True
        user_feature_type = 'data_matrix'

    elif not_unspecified(user_args.pyradigm_paths):
        user_feature_paths = list(map(abspath, user_args.pyradigm_paths))
        for pp in user_feature_paths:
            if not pexists(pp):
                raise IOError("One of pyradigms specified does not exist:\n {}".format(pp))

        atleast_one_feature_specified = True
        user_feature_type = 'pyradigm'
    else:
        user_feature_paths = None
        user_feature_type = None

    # map in python 3 returns a generator, not a list, so len() wouldnt work
    if not isinstance(user_feature_paths, list):
        user_feature_paths = list(user_feature_paths)

    if not atleast_one_feature_specified:
        raise ValueError('Atleast one method specifying features must be specified. '
                         'It can be a path(s) to pyradigm dataset, matrix file, user-defined folder or a Freesurfer subject directory.')

    return user_feature_paths, user_feature_type, fs_subject_dir


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser()

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        user_args = parser.parse_args()
    except:
        parser.exit(1)

    # noinspection PyUnboundLocalVariable
    meta_file = abspath(user_args.meta_file)
    if not pexists(meta_file):
        raise IOError("Meta data file doesn't exist.")

    user_feature_paths, user_feature_type, fs_subject_dir = organize_inputs(user_args)

    out_dir = realpath(user_args.out_dir)
    if not pexists(out_dir):
        try:
            os.mkdir(out_dir)
        except:
            raise IOError('Output folder could not be created.')

    train_perc = np.float32(user_args.train_perc)
    if not ( 0.01 <= train_perc <= 0.99):
        raise ValueError("Training percentage {} out of bounds - must be >= 0.01 and <= 0.99".format(train_perc))

    num_rep_cv = np.int64(user_args.num_rep_cv)
    if num_rep_cv < 10:
        raise ValueError("Atleast 10 repetitions of CV is recommened.")

    sample_ids, classes = get_metadata(meta_file)

    class_set, subgroups, positive_class = validate_class_set(classes, user_args.sub_groups, user_args.positive_class)

    feature_selection_size = validate_feature_selection_size(user_args.num_features_to_select)

    return sample_ids, classes, out_dir, \
           user_feature_paths, user_feature_type, fs_subject_dir, \
           train_perc, num_rep_cv, \
           positive_class, subgroups, feature_selection_size


def validate_feature_selection_size(feature_select_method, dim_in_data=None):
    """
    Ensures method chosen for the type of computation for the size of reduced dimensionality.

    Parameters
    ----------
    feature_select_method
    dim_in_data

    Returns
    -------

    """

    if feature_select_method.lower() in cfg.feature_selection_size_methods:
        num_select = feature_select_method
    elif feature_select_method.isdigit():
        num_select = np.int64(feature_select_method)
        if not 0 < num_select < np.Inf:
            raise UnboundLocalError('feature selection size out of bounds.\n'
                                    'Must be > 0 and < {}'.format(np.Inf))
    else:
        raise ValueError('Invalid choise - Choose an integer or one of \n{}'.format(cfg.feature_selection_size_methods))

    return num_select


def get_metadata(path):
    """
    Populates the dataset dictionary with subject ids and classes

    Currently supports the following per line: subjectid,class
    Future plans to include demographics data: subjectid,class,age,sex,education
    
    Returns
    -------
    sample_ids : list of str
        list of strings identifying the sample ids
    classes : dict
        dict of class labels for each id in the sample_ids

    """

    meta = np.genfromtxt(path, dtype=str, delimiter=cfg.DELIMITER)

    sample_ids = list(meta[:, 0])
    # checking for duplicates
    if len(set(sample_ids)) < len(sample_ids):
        duplicates = [sample for sample, count in Counter(sample_ids).items() if count > 1]
        raise ValueError('Duplicate sample ids found!\n{}\nRemove duplicates and rerun.'.format(duplicates))

    classes = dict(zip(sample_ids, meta[:, 1]))

    return sample_ids, classes


def get_dir_of_dirs(featdir, subjid):
    """
    Method to read in features for a given subject from a user-defined feature folder. This featdir must contain a
    separate folder for each subject with a file called features.txt with one number per line.
    
    Parameters
    ----------
    featdir : str
        Path to in the input directory
    subjid : str
        Subject id (name of the subfolder in the featdir identifying the subject)

    Returns
    -------
    data : ndarray 
        vector of features for the given subject
    feat_names : list of str
        names of each feature
        Currently None.
    """

    feat_names = None
    featfile = pjoin(featdir, subjid, 'features.txt')

    try:
        data = np.genfromtxt(featfile)
    except:
        raise IOError('Unable to load features from \n{}'.format(featfile))

    # the following ensures an array is returned even when data is a single scalar,
    # for which len() is not defined (which is needed for pyradigm to find dimensionality
    # order='F' (column-major) is chosen to as input is expected to be in a single column
    data = data.flatten(order='F')

    return data, feat_names


def get_data_matrix(featpath):
    "Returns ndarray from data matrix stored in a file"

    file_ext = os.path.splitext(featpath)[1].lower()
    try:
        if file_ext in ['.npy', '.numpy']:
            matrix = np.load(featpath)
        elif file_ext in ['.csv', '.txt']:
            matrix = np.loadtxt(featpath, delimiter=cfg.DELIMITER)
        else:
            raise ValueError(
                'Invalid or empty file extension : {}\n Allowed: {}'.format(file_ext, cfg.INPUT_FILE_FORMATS))
    except IOError:
        raise IOError('Unable to load the data matrix from disk.')
    except:
        raise

    return matrix


def get_pyradigm(feat_path):
    "Do-nothing reader of pyradigm."

    return feat_path


def get_features(subjects, classes, featdir, outdir, outname, get_method=None, feature_type='dir_of_dris'):
    """
    Populates the pyradigm data structure with features from a given method.

    Parameters
    ----------
    subjects : list or ndarray
        List of subject IDs
    classes : dict
        dict of class labels keyed in by subject id
    featdir : str
        Path to input directory to read the features from
    outdir : str
        Path to output directory to save the gathered features to.
    outname : str
        Name of the feature set
    get_method : callable
        Callable that takes in a path and returns a vectorized feature set (e.g. set of subcortical volumes),
        with an optional array of names for each feature.
    feature_type : str
        Identifier of data organization for features.

    Returns
    -------
    saved_path : str
        Path where the features have been saved to as an MLDataset

    """

    if not callable(get_method):
        raise ValueError("Supplied get_method is not callable! It must take in a path and return a vectorized feature set and labels.")

    # generating an unique numeric label for each class (sorted in order of their appearance in metadata file)
    class_set = set(classes.values())
    class_labels = dict()
    for idx, cls in enumerate(class_set):
        class_labels[cls] = idx

    ids_excluded = list()

    if feature_type == 'data_matrix':
        data_matrix = get_data_matrix(featdir)

    ds = MLDataset()
    for subjid in subjects:
        try:
            if feature_type == 'data_matrix':
                data = data_matrix[subjects.index(subjid), :]
                feat_names = None
            else:
                data, feat_names = get_method(featdir, subjid)

            ds.add_sample(subjid, data, class_labels[classes[subjid]], classes[subjid], feat_names)
        except:
            ids_excluded.append(subjid)
            traceback.print_exc()
            warnings.warn("Features for {} via {} method could not be read or added. "
                          "Excluding it.".format(subjid, get_method.__name__))

    # warning for if failed to extract features even for one subject
    alert_failed_feature_extraction(len(ids_excluded), ds.num_samples, len(subjects))

    # save the dataset to disk to enable passing on multiple dataset(s)
    saved_path = pjoin(outdir, outname)
    ds.save(saved_path)

    return saved_path


def alert_failed_feature_extraction(num_excluded, num_read, total_num):
    "Alerts user of failed feature extraction and get permission to proceed."

    allowed_to_proceed = True
    if num_excluded > 0:
        warnings.warn('Features for {} / {} subjects could not be read. '.format(num_excluded, total_num))
        user_confirmation = input("Would you like to proceed?  y / [N] : ")
        if user_confirmation.lower() not in ['y', 'yes', 'ye']:
            print('Stopping. \n'
                  'Rerun after completing the feature extraction for all subjects '
                  'or exclude failed subjects..')
            allowed_to_proceed = False  # unnecessary
            sys.exit(1)
        print(' Proceeding with only {} subjects.'.format(num_read))

    return allowed_to_proceed


def saved_dataset_matches(dataset_spec, subjects, classes):
    """
    Checks if the dataset on disk contains requested samples with the same classes

    Returns True only if the path to dataset exists, is not empy,
    contains the same number of samples, same sample ids and classes as in meta data!

    Parameters
    ----------
    dataset_spec : MLDataset or str
        dataset object, or path to one.
    subjects : list
        sample ids
    classes : dict
        class ids keyed in by sample ids

    Returns
    -------
    bool

    """

    if isinstance(dataset_spec, str):
        if (not pexists(dataset_spec)) or (os.path.getsize(dataset_spec) <= 0):
            return False
        else:
            ds = MLDataset(dataset_spec)
    elif isinstance(dataset_spec, MLDataset):
        ds = dataset_spec
    else:
        raise ValueError('Input must be a path or MLDataset.')

    if set(ds.class_set) != set(classes.values()) or set(ds.sample_ids) != set(subjects):
        return False
    else:
        return True


def make_visualizations(results_file_path, outdir):
    """
    Produces the performance visualizations/comparisons from the cross-validation results.
    
    Parameters
    ----------
    results_file_path : str
        Path to file containing results produced by `rhst`

    outdir : str
        Path to a folder to store results.

    """

    dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
    best_params, \
    feature_importances_rf, feature_names, \
    num_times_misclfd, num_times_tested, \
    confusion_matrix, class_order, accuracy_balanced, auc_weighted, positive_class = \
        rhst.load_results(results_file_path)

    if os.environ['DISPLAY'] is None:
        warnings.warn('DISPLAY is not set. Skipping to generate any visualizations.')
        return

    if not pexists(outdir):
        try:
            os.mkdir(outdir)
        except:
            raise IOError('Can not create output folder.')

    try:

        balacc_fig_path = pjoin(outdir, 'balanced_accuracy')
        visualize.metric_distribution(accuracy_balanced, method_names, balacc_fig_path,
                                      num_classes, "Balanced Accuracy")

        confmat_fig_path = pjoin(outdir, 'confusion_matrix')
        visualize.confusion_matrices(confusion_matrix, class_order, method_names, confmat_fig_path)

        cmp_misclf_fig_path = pjoin(outdir, 'compare_misclf_rates')
        if num_classes > 2:
            visualize.compare_misclf_pairwise(confusion_matrix, class_order, method_names, cmp_misclf_fig_path)
        elif num_classes == 2:
            visualize.compare_misclf_pairwise_parallel_coord_plot(confusion_matrix, class_order, method_names,
                                                                  cmp_misclf_fig_path)

        featimp_fig_path = pjoin(outdir, 'feature_importance')
        visualize.feature_importance_map(feature_importances_rf, method_names, featimp_fig_path, feature_names)

        misclf_out_path = pjoin(outdir, 'misclassified_subjects')
        visualize.freq_hist_misclassifications(num_times_misclfd, num_times_tested, method_names, misclf_out_path)
    except:
        traceback.print_exc()
        warnings.warn('Error generating the visualizations! Skipping ..')

    # cleaning up
    plt.close('all')

    return


def export_results(results_file_path, outdir):
    """
    Exports the results to simpler CSV format for use in other packages!
    
    Parameters
    ----------
    results_file_path
    outdir

    Returns
    -------
    None
    
    """

    dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
    best_params, \
    feature_importances_rf, feature_names, \
    num_times_misclfd, num_times_tested, \
    confusion_matrix, class_order, accuracy_balanced, auc_weighted, positive_class = \
        rhst.load_results(results_file_path)

    num_classes = confusion_matrix.shape[0]
    num_rep_cv = confusion_matrix.shape[2]
    num_datasets = confusion_matrix.shape[3]

    # separating CSVs from the PDFs
    exp_dir = pjoin(outdir, cfg.EXPORT_DIR_NAME)
    if not pexists(exp_dir):
        os.mkdir(exp_dir)

    # TODO think about how to export predictive probability per class per CV rep
    # pred_prob_per_class


    try:
        print('Saving accuracy distribution ..', end='')
        balacc_path = pjoin(exp_dir, 'balanced_accuracy.csv')
        np.savetxt(balacc_path, accuracy_balanced,
                   delimiter=cfg.DELIMITER,
                   fmt=cfg.EXPORT_FORMAT,
                   header=','.join(method_names))

        print('Done.')

        print('Saving confusion matrices ..', end='')
        cfmat_reshaped = np.reshape(confusion_matrix, [num_classes * num_classes, num_rep_cv, num_datasets])
        for mm in range(num_datasets):
            confmat_path = pjoin(exp_dir, 'confusion_matrix_{}.csv'.format(method_names[mm]))
            np.savetxt(confmat_path,
                       cfmat_reshaped[:, :, mm].T,  # NOTICE the transpose
                       delimiter=cfg.DELIMITER, fmt=cfg.EXPORT_FORMAT,
                       comments='shape of confusion matrix: num_repetitions x num_classes^2')
        print('Done.')

        print('Saving misclassfiication rates ..', end='')
        avg_cfmat, misclf_rate = visualize.compute_pairwise_misclf(confusion_matrix)
        num_datasets = misclf_rate.shape[0]
        for mm in range(num_datasets):
            cmp_misclf_path = pjoin(exp_dir, 'average_misclassification_rates_{}.csv'.format(method_names[mm]))
            np.savetxt(cmp_misclf_path,
                       misclf_rate[mm, :],
                       fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER)
        print('Done.')

        print('Saving feature importance values ..', end='')
        for mm in range(num_datasets):
            featimp_path = pjoin(exp_dir, 'feature_importance_{}.csv'.format(method_names[mm]))
            np.savetxt(featimp_path,
                       feature_importances_rf[mm],
                       fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER,
                       header=','.join(feature_names[mm]))
        print('Done.')

        print('Saving subject-wise misclassification frequencies ..', end='')
        perc_misclsfd, _, _, _ = visualize.compute_perc_misclf_per_sample(num_times_misclfd, num_times_tested)
        for mm in range(num_datasets):
            subwise_misclf_path = pjoin(exp_dir, 'subject_misclf_freq_{}.csv'.format(method_names[mm]))
            # TODO there must be a more elegant way to write dict to CSV
            with open(subwise_misclf_path, 'w') as smf:
                for sid, val in perc_misclsfd[mm].items():
                    smf.write('{}{}{}\n'.format(sid, cfg.DELIMITER, val))
        print('Done.')

    except:
        traceback.print_exc()
        raise IOError('Unable to export the results to CSV files.')

    return


def validate_class_set(classes, subgroups, positiveclass=None):
    "Ensures class names are valid and sub-groups exist."

    class_set = set(classes.values())

    if subgroups != 'all':
        for comb in subgroups:
            cls_list = comb.split(',')
            # ensuring each subgroup has atleast two classes
            if len(set(cls_list)) < 2:
                raise ValueError('This subgroup {} does not contain two unique classes.'.format(comb))

            # verify each of them were defined in meta
            for cls in cls_list:
                if cls not in class_set:
                    raise ValueError("Class {} in combination {} "
                                     "does not exist in meta data.".format(cls, comb))
    else:
        # using all classes
        subgroups = ','.join(class_set)

    # the following loop is required to preserve original order
    # this does not: class_order_in_meta = list(set(classes.values()))
    class_order_in_meta = list()
    for x in class_set:
        if x not in class_order_in_meta:
            class_order_in_meta.append(x)

    num_classes = len(class_order_in_meta)
    if num_classes < 2:
        raise ValueError("Atleast two classes are required for predictive analysis! "
                         "Only one given ({})".format(set(classes.values())))

    if num_classes == 2:
        if not_unspecified(positiveclass):
            if positiveclass not in class_order_in_meta:
                raise ValueError('Positive class specified does not exist in meta data.\n'
                                 'Choose one of {}'.format(class_order_in_meta))
            print('Positive class specified for AUC calculation: {}'.format(positiveclass))
        else:
            positiveclass = class_order_in_meta[-1]
            print('Positive class inferred for AUC calculation: {}'.format(positiveclass))

    return class_set, subgroups, positiveclass


def make_dataset_filename(method_name):
    "File name constructor."

    file_name = 'consolidated_{}_{}.MLDataset.pkl'.format(method_name, make_time_stamp())

    return file_name


def import_datasets(method_list, out_dir, subjects, classes, feature_path, feature_type='dir_of_dirs'):
    """
    Imports all the specified feature sets and organizes them into datasets.
     
    Parameters
    ----------
    method_list : list of callables
        Set of predefined methods returning a vector of features for a given sample id and location
    out_dir : str
        Path to the output folder

    subjects : list of str
        List of sample ids
    classes : dict
        Dict identifying the class for each sample id in the dataset.
    feature_path : list of str
        List of paths to the root directory containing the features (pre- or user-defined).
        Must be of same length as method_list
    feature_type : str
        a string identifying the structure of feature set.
        Choices = ('dir_of_dirs', 'data_matrix')
        
    Returns
    -------    
    method_names : list of str
        List of method names used for annotation.
    dataset_paths_file : str
        Path to the file containing paths to imported feature sets.
    
    """

    method_names = list()
    outpath_list = list()
    for mm, cur_method in enumerate(method_list):
        if cur_method in [get_dir_of_dirs]:
            method_name = basename(feature_path[mm])
        elif cur_method in [get_data_matrix]:
            method_name = os.path.splitext(basename(feature_path[mm]))[0]
        elif cur_method in [get_pyradigm]:
            loaded_dataset = MLDataset(feature_path[mm])
            if len(loaded_dataset.description) > 1:
                method_name = loaded_dataset.description.replace(' ', '_')
            else:
                method_name = basename(feature_path[mm])
            method_names.append(method_name)
            if saved_dataset_matches(loaded_dataset, subjects, classes):
                outpath_list.append(feature_path[mm])
                continue
            else:
                raise ValueError('supplied pyradigm dataset does not match samples in the meta data.')
        else:
            # adding an index for an even more unique identification
            # method_name = '{}_{}'.format(cur_method.__name__,mm)
            method_name = cur_method.__name__

        method_names.append(method_name)
        out_name = make_dataset_filename(method_name)

        outpath_dataset = pjoin(out_dir, out_name)
        if not saved_dataset_matches(outpath_dataset, subjects, classes):
            # noinspection PyTypeChecker
            outpath_dataset = get_features(subjects, classes,
                                           feature_path[mm],
                                           out_dir, out_name,
                                           cur_method, feature_type)

        outpath_list.append(outpath_dataset)

    combined_name = uniq_combined_name(method_names)

    dataset_paths_file = pjoin(out_dir, 'datasetlist.' + combined_name + '.txt')
    with open(dataset_paths_file, 'w') as dpf:
        dpf.writelines('\n'.join(outpath_list))

    return method_names, dataset_paths_file


def uniq_combined_name(method_names, max_len=180, num_char_each_word=1):
    "Function to produce a uniq, and not a long combined name. Recursive"

    re_delimiters_word = '_|; |, |\*|\n'
    combined_name = '_'.join(method_names)
    # depending on number and lengths of method_names, this can get very long
    if len(combined_name) > max_len:
        first_letters = list()
        for mname in method_names:
            first_letters.append(''.join([word[:num_char_each_word] for word in re.split(re_delimiters_word, mname)]))
        combined_name = '_'.join(first_letters)

        if len(combined_name) > max_len:
            combined_name = uniq_combined_name(first_letters)

    return combined_name


def make_method_list(fs_subject_dir, user_feature_paths, user_feature_type='dir_of_dirs'):
    """
    Returns an organized list of feature paths and methods to read in features.
    
    Parameters
    ----------
    fs_subject_dir : str
    user_feature_paths : list of str
    user_feature_type : str

    Returns
    -------
    feature_dir : list
    method_list : list
    

    """

    freesurfer_readers = [aseg_stats_subcortical, aseg_stats_whole_brain]
    userdefined_readers = {'dir_of_dirs': get_dir_of_dirs,
                           'data_matrix': get_data_matrix,
                           'pyradigm': get_pyradigm}

    feature_dir = list()
    method_list = list()
    if not_unspecified(user_feature_paths):
        if user_feature_type not in userdefined_readers:
            raise NotImplementedError("Invalid feature type or its reader is not implemented yet!")

        for upath in user_feature_paths:
            feature_dir.append(upath)
            method_list.append(userdefined_readers[user_feature_type])

    if not_unspecified(fs_subject_dir):
        for fsrdr in freesurfer_readers:
            feature_dir.append(fs_subject_dir)
            method_list.append(fsrdr)

    if len(method_list) != len(feature_dir):
        raise ValueError('Invalid specification for features!')

    if len(method_list) < 1:
        raise ValueError('Atleast one feature set must be specified.')

    print("\nRequested features for analysis:")
    for mm, method in enumerate(method_list):
        print("{} from {}".format(method.__name__, feature_dir[mm]))

    return feature_dir, method_list


def run_cli():
    """
    Main entry point.
    
    """

    # TODO design an API interface for advanced access as an importable package

    subjects, classes, out_dir, user_feature_paths, user_feature_type, \
        fs_subject_dir, train_perc, num_rep_cv, positiveclass, subgroups, \
        feature_selection_size = parse_args()

    feature_dir, method_list = make_method_list(fs_subject_dir, user_feature_paths, user_feature_type)

    # TODO need to be able to parallelize at subgroup- and method-level
    method_names, dataset_paths_file = import_datasets(method_list, out_dir, subjects, classes,
                                                       feature_dir, user_feature_type)

    results_file_path = rhst.run(dataset_paths_file, method_names, out_dir,
                                 train_perc=train_perc, num_repetitions=num_rep_cv,
                                 positive_class=positiveclass,
                                 feat_sel_size=feature_selection_size)

    print('Saving the visualizations and results to \n{}'.format(out_dir))
    make_visualizations(results_file_path, out_dir)

    # TODO avoid loading results from disk twice (vis & export)
    export_results(results_file_path, out_dir)

    return


def fit(feature_sets, meta_data, output_dir,
        pipeline=None,
        train_perc=0.5,
        num_repetitions=200,
        positive_class=None,
        feat_sel_size=cfg.default_num_features_to_select):
    """
    Generate comprehensive report on the predictive performance for different feature sets and statistically compare them.

    Main entry point for API access.

    Parameters
    ----------
    feature_sets : multiple
        The input can be specified in either of the following ways:

            - path to a file containing list of paths (each line containing path to a valid MLDataset)
            - list of paths to MLDatasets saved on disk
            - list of MLDatasets
            - list of tuples (to specify multiple features), each element containing (X, y) i.e. data and target labels
            - a single tuple containing (X, y) i.e. data and target labels
            - list of paths to CSV files, each containing one type of features.

            When specifying multiple sets of input features, ensure:
            - all of them contain the same number of samples
            - each sample belongs to same class across all feature sets.

    meta_data : multiple
        The meta data can be specified in either of the following ways:

            - a path to a meta data file (see :doc:`features` page)
            - a dict keyed in by sample IDs with values representing their classes.
            - None, if meta data is already specified in ``feature_sets`` input.

    pipeline : object
        A sciki-learn pipeline describing the sequence of steps. This is typically a set of feature selections or dimensionality reduction steps followed by an estimator (classifier).

        See http://scikit-learn.org/stable/modules/pipeline.html#pipeline for more details.

        Default: None, which leads to the selection of a Random Forest classifier with feature selection based on mutual information (top k = N_train/10 features).

    method_names : list
        A list of names to denote the different feature sets

    out_results_dir : str
        Path to output directory to save the cross validation results to.

    train_perc : float, optional
        Percetange of subjects to train the classifier on.
        The percentage is applied to the size of the smallest class to estimate
        the number of subjects from each class to be reserved for training.
        The smallest class is chosen to avoid class-imbalance in the training set.
        Default: 0.8 (80%).

    num_repetitions : int, optional
        Number of repetitions of cross-validation estimation. Default: 200.

    positive_class : str
        Name of the class to be treated as positive in calculation of AUC

    feat_sel_size : str or int
        Number of features to retain after feature selection.
        Must be a method (tenth or square root of the size of smallest class in training set,
            or a finite integer smaller than the data dimensionality.

    Returns
    -------
    results_path : str
        Path to pickle file containing full set of CV results.

    """

    raise NotImplementedError

    return


if __name__ == '__main__':
    run_cli()
