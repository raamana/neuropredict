#/usr/bin/python

__all__ = ['run', 'get_parser' ]

import argparse
import sys
import traceback
import warnings
from collections import Counter
from time import localtime, strftime
from sys import version_info
from os.path import join as pjoin, exists as pexists, abspath, realpath

from pyradigm import MLDataset

if version_info.major==2 and version_info.minor==7:
    import rhst, visualize
    from freesurfer import *
    import config_neuropredict as cfg
elif version_info.major > 2:
    from neuropredict import rhst, visualize
    from neuropredict.freesurfer import *
    from neuropredict import config_neuropredict as cfg
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')


def make_time_stamp():
    "Returns a timestamp string."

    # just by the hour
    return strftime('%Y%m%d-T%H', localtime())

def not_unspecified( var ):
    """ Checks for null values of a give variable! """

    return var not in [ 'None', None, '' ]


def get_parser():
    "Parser to specify arguments and their defaults."

    parser = argparse.ArgumentParser(prog="neuropredict")

    parser.add_argument("-m", "--metadatafile", action="store", dest="metadatafile",
                        default=None, required=True,
                        help="Abs path to file containing metadata for subjects to be included for analysis. At the "
                             "minimum, each subject should have an id per row followed by the class it belongs to. "
                             "E.g. \n"
                             "sub001,control\n"
                             "sub002,control\n"
                             "sub003,disease\n"
                             "sub004,disease\n")

    parser.add_argument("-o", "--outdir", action="store", dest="outdir",
                        required=True,
                        help="Output folder to store features and results.")

    parser.add_argument("-f", "--fsdir", action="store", dest="fsdir",
                        default=None,
                        help="Absolute path to SUBJECTS_DIR containing the finished runs of Freesurfer parcellation"
                             " (each subject named after its ID in the metadata file). "
                             "\nE.g. --fsdir /project/freesurfer_v5.3")

    user_defined = parser.add_mutually_exclusive_group()
    user_defined.add_argument("-u", "--user_feature_paths", action="store", dest="user_feature_paths",
                              nargs = '+', # to allow for multiple features
                              default=None,
                              help="List of absolute paths to an user's own features."
                                 "\nEach folder contains a separate folder for each subject "
                                 "  (named after its ID in the metadata file) "
                                 "containing a file called features.txt with one number per line.\n"
                                 "All the subjects (in a given folder) must have the number of features (#lines in file). "
                                 "Different folders can have different number of features for each subject."
                                 "\n Names of each folder is used to annotate the results in visualizations. "
                                 "Hence name them uniquely and meaningfully, keeping in mind these figures will be included in your papers."
                                 "\nE.g. --user_feature_paths /project/fmri/ /project/dti/ /project/t1_volumes/ ."
                                 " \n Only one of user_feature_paths and user_feature_paths options can be specified.")

    user_defined.add_argument("-d", "--data_matrix_path", action="store", dest="data_matrix_path",
                              nargs = '+',
                              default=None,
                              help="List of absolute paths to text files containing one matrix "
                                 " of size N x p (num_samples x num_features). "
                                 " Each row in the data matrix file must represent data corresponding "
                                 " to sample in the same row of the meta data file "
                                   "(meta data file and data matrix must be in row-wise correspondence). "
                                 "Name of this file will be used to annotate the results and visualizations."
                                 "\nE.g. --data_matrix_path /project/fmri.csv /project/dti.csv /project/t1_volumes.csv. "
                                 " \n Only one of user_feature_paths and user_feature_paths options can be specified."
                                 "File format could be 1) a simple comma-separated text file (with extension .csv or .txt): "
                                   "which can easily be read back with numpy.loadtxt(filepath, delimiter=',') or "
                                   "2) a numpy array saved to disk (with extension .npy or .numpy) that can read in with numpy.load(filepath). "
                                   "One could use numpy.savetxt(data_array, delimiter=',') or numpy.save(data_array) to save features."
                                   "File format is inferred from its extension.")

    parser.add_argument("-p", "--positiveclass", action="store", dest="positiveclass",
                        default=None,
                        help="Name of the positive class (Alzheimers, MCI or Parkinsons etc) "
                             "to be used in calculation of area under the ROC curve. "
                             "Applicable only for binary classification experiments. "
                             "Default: class appearning second in order specified in metadata file.")

    parser.add_argument("-t", "--trainperc", action="store", dest="train_perc",
                        default=0.5,
                        help="Percentage of the smallest class to be reserved for training. "
                             "Must be in the interval [0.01 0.99]."
                             "If sample size is sufficiently big, we recommend 0.5."
                             "If sample size is small, or class imbalance is high, choose 0.8.")

    parser.add_argument("-n", "--numrep", action="store", dest="num_rep_cv",
                        default=200,
                        help="Number of repetitions of the repeated-holdout cross-validation. "
                             "The larger the number, the better the estimates will be.")

    parser.add_argument("-a", "--atlas", action="store", dest="atlasid",
                        default="fsaverage",
                        help="Name of the atlas to use for visualization."
                             "\nDefault: fsaverage, if available.")

    parser.add_argument("-s", "--subgroup", action="store", dest="subgroup",
                        nargs="*",
                        default="all",
                        help="This option allows the user to study different combinations of classes in multi-class (N>2) dataset. "
                             "For example, in a dataset with 3 classes CN, FTD and AD, two studies of pair-wise combinations can be studied"
                             " with the following flag --subgroup CN,FTD CN,AD . "
                             "This allows the user to focus on few interesting subgroups depending on their dataset/goal. "
                             "Format: each subgroup must be a comma-separated list of classes. "
                             "Hence it is strongly recommended to use class names without any spaces, commas, hyphens and special characters, and "
                             "ideally just alphanumeric characters separated by underscores. "
                             "Default: all - using all the available classes in a all-vs-all multi-class setting.")

    return parser


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser()

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        options = parser.parse_args()
    except:
        parser.exit(1)

    # noinspection PyUnboundLocalVariable
    metadatafile = abspath(options.metadatafile)
    assert pexists(metadatafile), "Given metadata file doesn't exist."

    atleast_one_feature_specified = False
    if not_unspecified(options.fsdir):
        fsdir = abspath(options.fsdir)
        assert pexists(fsdir), "Given Freesurfer directory doesn't exist."
        atleast_one_feature_specified = True
    else:
        fsdir = None

    if not_unspecified(options.user_feature_paths):
        user_feature_paths = map(abspath, options.user_feature_paths)
        for udir in user_feature_paths:
            assert pexists(udir), "One of the user directories for features doesn't exist:" \
                                         "\n {}".format(udir)

        atleast_one_feature_specified = True
        user_feature_type = 'dir_of_dirs'

    elif not_unspecified(options.data_matrix_path):
        user_feature_paths = map(abspath, options.data_matrix_path)
        for dm in user_feature_paths:
            assert pexists(dm), "One of the data matrices specified does not exist:\n {}".format(dm)

        atleast_one_feature_specified = True
        user_feature_type = 'data_matrix'
    else:
        user_feature_paths = None
        user_feature_type  = None

    if not atleast_one_feature_specified:
        raise ValueError('Atleast a Freesurfer directory or one user-defined directory or matrix must be specified.')

    outdir = abspath(options.outdir)
    if not pexists(outdir):
        try:
            os.mkdir(outdir)
        except:
            raise

    train_perc = np.float32(options.train_perc)
    assert (train_perc >= 0.01 and train_perc <= 0.99), \
        "Training percentage {} out of bounds - must be > 0.01 and < 0.99".format(train_perc)

    num_rep_cv = np.int64(options.num_rep_cv)
    assert num_rep_cv >= 10, \
        "Atleast 10 repetitions of CV is recommened."

    sample_ids, classes = get_metadata(metadatafile)
    class_set = set(classes.values())

    subgroups = options.subgroup
    if subgroups != 'all':
        for comb in subgroups:
            for cls in comb.split(','):
                assert cls in class_set, \
                    "Class {} in combination {} does not exist in meta data.".format(cls, comb)
    else:
        # using all classes
        subgroups = ','.join(class_set)


    return sample_ids, classes, outdir, \
           user_feature_paths, user_feature_type, \
           fsdir, \
           train_perc, num_rep_cv, \
           options.positiveclass, subgroups


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

    # sample_ids = list()
    # classes = dict()
    # with open(path) as mf:
    #     for line in mf:
    #         if not line.startswith('#'):
    #             parts = line.strip().split(',')
    #             sid = parts[0]
    #             sample_ids.append(sid) # maintaining order of appearance is important as we use data_matrix input mechanism
    #             classes[sid] = parts[1]

    meta = np.genfromtxt(path, dtype=str, delimiter=cfg.DELIMITER)

    sample_ids = list(meta[:,0])
    # checking for duplicates
    if len(set(sample_ids)) < len(sample_ids):
        duplicates = [ sample for sample, count in Counter(sample_ids).items() if count > 1 ]
        raise ValueError('Duplicate sample ids found!\n{}\nRemove duplicates and rerun.'.format(duplicates))

    classes    = dict(zip(sample_ids, meta[:,1]))

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
            raise ValueError('Invalid or empty file extension : {}\n Allowed: {}'.format(file_ext, cfg.INPUT_FILE_FORMATS))
    except IOError:
        raise IOError('Unable to load the data matrix from disk.')
    except:
        raise

    return matrix


def get_features(subjects, classes, featdir, outdir, outname, getmethod = None, feature_type ='dir_of_dris'):
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
    getmethod : callable
        Callable that takes in a path and returns a vectorized feature set (e.g. set of subcortical volumes),
        with an optional array of names for each feature.
    feature_type : str
        Identifier of data organization for features.

    Returns
    -------
    saved_path : str
        Path where the features have been saved to as an MLDataset

    """

    assert callable(getmethod), "Supplied getmethod is not callable!" \
                                "It must take in a path and return a vectorized feature set and labels."

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
                data = data_matrix[subjects.index(subjid),:]
                feat_names = None
            else:
                data, feat_names = getmethod(featdir, subjid)

            ds.add_sample(subjid, data, class_labels[classes[subjid]], classes[subjid], feat_names)
        except:
            ids_excluded.append(subjid)
            traceback.print_exc()
            warnings.warn("Features for {} via {} method could not be read or added. "
                          "Excluding it.".format(subjid, getmethod.__name__))

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
            print ('Stopping. \n'
                          'Rerun after completing the feature extraction for all subjects '
                          'or exclude failed subjects..')
            allowed_to_proceed = False # unnecessary
            sys.exit(1)
        print(' Proceeding with only {} subjects.'.format(num_read))

    return allowed_to_proceed


def saved_dataset_matches(ds_path, subjects, classes):
    """
    Returns True only if the path to dataset
        exists, is not empy,
        contains the same number of samples,
        same sample ids and classes as in meta data!

    :returns bool.
    """

    if (not pexists(ds_path)) or (os.path.getsize(ds_path) <= 0):
        return False
    else:
        ds = MLDataset(ds_path)
        if set(ds.class_set) != set(classes) or set(ds.sample_ids) != set(subjects):
            return False
        else:
            return True


def visualize_results(results_file_path, outdir, method_names):
    """
    Produces the performance visualizations/comparisons from the cross-validation results.
    
    Parameters
    ----------
    results_file_path
    outdir
    method_names

    """

    dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
        pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
        best_min_leaf_size, best_num_predictors, \
        feature_importances_rf, feature_names, \
        num_times_misclfd, num_times_tested, \
        confusion_matrix, class_order, accuracy_balanced, auc_weighted, positive_class = \
            rhst.load_results(results_file_path)

    if os.environ['DISPLAY'] is None:
        warnings.warn('DISPLAY is not set. Skipping to generate any visualizations.')
        return

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
            visualize.compare_misclf_pairwise_parallel_coord_plot(confusion_matrix, class_order, method_names, cmp_misclf_fig_path)

        featimp_fig_path = pjoin(outdir, 'feature_importance')
        visualize.feature_importance_map(feature_importances_rf, method_names, featimp_fig_path, feature_names)

        misclf_out_path = pjoin(outdir, 'misclassified_subjects')
        visualize.freq_hist_misclassifications(num_times_misclfd, num_times_tested, method_names, misclf_out_path)
    except:
        traceback.print_exc()
        warnings.warn('Error generating the visualizations! Skipping ..')



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
        best_min_leaf_size, best_num_predictors, \
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
        balacc_path = pjoin(exp_dir, 'balanced_accuracy.csv')
        np.savetxt(balacc_path, accuracy_balanced,
                   delimiter=cfg.DELIMITER,
                   fmt = cfg.EXPORT_FORMAT,
                   header=','.join(method_names))

        cfmat_reshaped = np.reshape(confusion_matrix, [num_classes*num_classes, num_rep_cv, num_datasets] )
        for mm in range(num_datasets):
            confmat_path = pjoin(exp_dir, 'confusion_matrix_{}.csv'.format(method_names[mm]))
            np.savetxt(confmat_path,
                       cfmat_reshaped[:,:,mm].T, # NOTICE the transpose
                       delimiter=cfg.DELIMITER, fmt=cfg.EXPORT_FORMAT,
                       comments= 'shape of confusion matrix: num_repetitions x num_classes^2')

        avg_cfmat, misclf_rate = visualize.compute_pairwise_misclf(confusion_matrix)
        num_datasets = misclf_rate.shape[0]
        for mm in range(num_datasets):
            cmp_misclf_path = pjoin(exp_dir, 'average_misclassification_rates_{}.csv'.format(method_names[mm]))
            np.savetxt(cmp_misclf_path,
                       misclf_rate[mm,:],
                       fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER)

        for mm in range(num_datasets):
            featimp_path = pjoin(exp_dir, 'feature_importance_{}.csv'.format(method_names[mm]))
            np.savetxt(featimp_path,
                       feature_importances_rf[mm],
                       fmt=cfg.EXPORT_FORMAT, delimiter=cfg.DELIMITER,
                       header=','.join(feature_names[mm]))

        perc_misclsfd, _, _, _ = visualize.compute_perc_misclf_per_sample(num_times_misclfd, num_times_tested)
        for mm in range(num_datasets):
            subwise_misclf_path = pjoin(exp_dir, 'subject_misclf_freq_{}.csv'.format(method_names[mm]))
            # TODO there must be a more elegant way to write dict to CSV
            with open(subwise_misclf_path, 'w') as smf:
                for sid, val in perc_misclsfd[mm].items():
                    smf.write('{}{}{}\n'.format(sid, cfg.DELIMITER, val))

    except:
        traceback.print_exc()
        raise IOError('Unable to export the results to CSV files.')

    return


def validate_class_set(classes, positiveclass=None):
    ""

    # the following loop is required to preserve original order
    # this does not: class_set_in_meta = list(set(classes.values()))
    class_set_in_meta = list()
    for x in classes.values():
        if x not in class_set_in_meta:
            class_set_in_meta.append(x)

    num_classes = len(class_set_in_meta)
    if num_classes < 2:
        raise ValueError("Atleast two classes are required for predictive analysis! "
                         "Only one given ({})".format(set(classes.values())))

    if num_classes == 2:
        if not_unspecified(positiveclass):
            if positiveclass not in class_set_in_meta:
                raise ValueError('Positive class specified does not exist in meta data.\n'
                                 'Choose one of {}'.format(class_set_in_meta))
            print('Positive class specified for AUC calculation: {}'.format(positiveclass))
        else:
            positiveclass = class_set_in_meta[-1]
            print('Positive class inferred for AUC calculation: {}'.format(positiveclass))

    return positiveclass


def make_dataset_filename(method_name):
    "File name constructor."

    file_name = 'consolidated_{}_{}.MLDataset.pkl'.format(method_name, make_time_stamp())

    return file_name


def import_datasets(method_list, outdir, subjects, classes, feature_path, feature_type='dir_of_dirs'):
    """
    Imports all the specified feature sets and organizes them into datasets.
     
    Parameters
    ----------
    method_list : list of callables
        Set of predefined methods returning a vector of features for a given sample id and location
    outdir : str
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
        if cur_method in [ get_dir_of_dirs ]:
            method_name = os.path.basename(feature_path[mm])
        elif cur_method in [ get_data_matrix ]:
            method_name = os.path.splitext(os.path.basename(feature_path[mm]))[0]
        else:
            # adding an index for an even more unique identification
            # method_name = '{}_{}'.format(cur_method.__name__,mm)
            method_name = cur_method.__name__

        method_names.append(method_name)
        out_name = make_dataset_filename(method_name)

        outpath_dataset = pjoin(outdir, out_name)
        if not saved_dataset_matches(outpath_dataset, subjects, classes):
            # noinspection PyTypeChecker
            outpath_dataset = get_features(subjects, classes,
                                           feature_path[mm],
                                           outdir, out_name,
                                           cur_method, feature_type)

        outpath_list.append(outpath_dataset)

    combined_name = '_'.join(method_names)
    dataset_paths_file = pjoin(outdir, 'datasetlist.' + combined_name+ '.txt')
    with open(dataset_paths_file, 'w') as dpf:
        dpf.writelines('\n'.join(outpath_list))

    return method_names, dataset_paths_file


def make_method_list(fsdir, user_feature_paths, user_feature_type='dir_of_dirs'):
    """
    Returns an organized list of feature paths and methods to read in features.
    
    Parameters
    ----------
    fsdir : str
    user_feature_paths : list of str
    user_feature_type : str

    Returns
    -------
    feature_dir : list
    method_list : list
    

    """

    freesurfer_readers = [aseg_stats_subcortical, aseg_stats_whole_brain]
    userdefined_readers= { 'dir_of_dirs': get_dir_of_dirs,
                           'data_matrix': get_data_matrix }

    feature_dir = list()
    method_list = list()
    if not_unspecified(user_feature_paths):
        if user_feature_type not in userdefined_readers:
            raise NotImplementedError("Invalid feature type or its reader is not implemented yet!")

        for upath in user_feature_paths:
            feature_dir.append(upath)
            method_list.append(userdefined_readers[user_feature_type])

    if not_unspecified(fsdir):
        for fsrdr in freesurfer_readers:
            feature_dir.append(fsdir)
            method_list.append(fsrdr)

    if len(method_list) != len(feature_dir):
        raise ValueError('Invalid specification for features!')

    if len(method_list) < 1:
        raise ValueError('Atleast one feature set must be specified.')

    print("\nRequested features for analysis:")
    for mm, method in enumerate(method_list):
        print("{} from {}".format(method.__name__, feature_dir[mm]))

    return feature_dir, method_list


def run():
    """
    Main entry point.
    
    """

    # TODO design an API interface for advanced access as an importable package

    subjects, classes, outdir, user_feature_paths, user_feature_type, \
        fsdir, train_perc, num_rep_cv, positiveclass, subgroups = parse_args()

    positiveclass = validate_class_set(classes, positiveclass)

    feature_dir, method_list = make_method_list(fsdir, user_feature_paths, user_feature_type)

    # TODO need to be able to parallelize at subgroup- and method-level
    method_names, dataset_paths_file = import_datasets(method_list, outdir, subjects, classes,
                                                       feature_dir, user_feature_type)

    results_file_path = rhst.run(dataset_paths_file, method_names, outdir,
                                 train_perc=train_perc, num_repetitions=num_rep_cv,
                                 positive_class= positiveclass)

    visualize_results(results_file_path, outdir, method_names)

    export_results(results_file_path, outdir)

if __name__ == '__main__':
    run()
