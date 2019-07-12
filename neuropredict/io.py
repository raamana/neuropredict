
import os
import sys
import traceback
import warnings
from collections import Counter
from os.path import join as pjoin, exists as pexists, realpath, basename
import numpy as np
from neuropredict import config_neuropredict as cfg
from neuropredict.utils import make_dataset_filename
from pyradigm import MLDataset

def get_metadata_in_pyradigm(meta_data_supplied, meta_data_format='pyradigm'):
    "Returns sample IDs and their classes from a given pyradigm"

    meta_data_format = meta_data_format.lower()
    if meta_data_format in ['pyradigm', 'mldataset']:
        dataset = MLDataset(filepath=realpath(meta_data_supplied))
    elif meta_data_format in ['arff', 'weka']:
        dataset = MLDataset(arff_path=realpath(meta_data_supplied))
    else:
        raise NotImplementedError('Meta data format {} not implemented. '
                                  'Only pyradigm and ARFF are supported.'.format(meta_data_format))

    return dataset.sample_ids, dataset.classes


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


def process_pyradigm(feature_path, subjects, classes):
    """Processes the given dataset to return a clean name and path."""

    loaded_dataset = MLDataset(filepath=feature_path)

    if len(loaded_dataset.description) > 1:
        method_name = loaded_dataset.description
    else:
        method_name = basename(feature_path)

    if not saved_dataset_matches(loaded_dataset, subjects, classes):
        raise ValueError('supplied pyradigm dataset does not match samples in the meta data.')
    else:
        out_path_cur_dataset = feature_path

    return method_name, out_path_cur_dataset


def get_arff(feat_path):
    "Do-nothing reader for ARFF format."

    return feat_path


def process_arff(feature_path, subjects, classes, out_dir):
    """Processes the given dataset to return a clean name and path."""


    loaded_dataset = MLDataset(arff_path=feature_path)
    if len(loaded_dataset.description) > 1:
        method_name = loaded_dataset.description
    else:
        method_name = basename(feature_path)

    out_name = make_dataset_filename(method_name)
    out_path_cur_dataset = pjoin(out_dir, out_name)
    loaded_dataset.save(out_path_cur_dataset)

    if not saved_dataset_matches(loaded_dataset, subjects, classes):
        raise ValueError('supplied ARFF dataset does not match samples in the meta data.')

    return method_name, out_path_cur_dataset


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
        raise ValueError("Supplied get_method is not callable! "
                         "It must take in a path and return a vectorized feature set and labels.")

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
    saved_path = realpath(pjoin(outdir, outname))
    try:
        ds.save(saved_path)
    except IOError as ioe:
        print('Unable to save {} features to disk in folder:\n{}'.format(outname, outdir))
        raise ioe

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

    # TODO this check for exact match is too conservative: allow for extra subjects/classes to exist
    #   as long as they contain the subjects you need, and they belong to right class in both
    if set(ds.class_set) != set(classes.values()) or set(ds.sample_ids) != set(subjects):
        return False
    else:
        return True


def load_pyradigms(dataset_paths, sub_group=None):
    """Reads in a list of datasets in pyradigm format.

    Parameters
    ----------
    dataset_paths : iterable
        List of paths to pyradigm dataset

    sub_group : iterable
        subset of classes to return. Default: return all classes.
        If sub_group is specified, returns only that subset of classes for all datasets.

    Raises
    ------
        ValueError
            If all the datasets do not contain the request subset of classes.

    """

    if sub_group is not None:
        sub_group = set(sub_group)

    # loading datasets
    datasets = list()
    for fp in dataset_paths:
        if not pexists(fp):
            raise IOError("Dataset @ {} does not exist.".format(fp))

        try:
            # there is an internal validation of dataset
            ds_in = MLDataset(fp)
        except:
            print("Dataset @ {} is not a valid MLDataset!".format(fp))
            raise

        class_set = set(ds_in.class_set)
        if sub_group is None or sub_group == class_set:
            ds_out = ds_in
        elif sub_group < class_set: # < on sets is an issubset operation
            ds_out = ds_in.get_class(sub_group)
        else:
            raise ValueError('One or more classes in {} does not exist in\n{}'.format(sub_group, fp))

        # add the valid dataset to list
        datasets.append(ds_out)

    return datasets

