
import os
import numpy as np
import random
import sklearn
from sklearn.ensemble import RandomForestClassifier

from pyradigm import MLDataset

def clf_tuner(train_fs, test_fs):
    "Method to optimize the classifier on the training set and returns predictions on test set. "

    pass


def calc_balanced_accuracy(confmat):
    "Computes the balanced accuracy in a given confusion matrix!"
    # IMPLEMENT TODO



def run(dataset_paths, out_results_dir, train_perc = 0.8, num_repetitions = 200):
    """

    :param dataset_paths: path to file containing list of paths (each containing a valid MLDataset).
    :param out_results_dir: path to save the results to (in various forms)
    :param train_perc: percetange of subjects to train the classifier on. The percentage is applied to the size of
    the smallest class to estimate the numner of subjects from each class to be reserved for training. The smallest
    class is chosen to avoid class-imbalance in the training set. Default: 0.8 (80%).
    :param num_repetitions: number of repetitions of cross-validation estimation. Default: 200.
    :return:
    """

    # load datasets
    # validate each dataset
    # ensure same number of subjects across all datasets
    #        same number of features/subject in each dataset
    #        same class set across all datasets
    # re-map the labels (from 1 to n) to ensure numeric labels do not differ
    # sort them if need be (not needed if MLDatasets)
    # for rep 1 to N, for feat 1 to M,
    #   run train/test/evaluate.
    #   keep tab on misclassifications
    # save results (comprehensive and reloadable manner)

    # loading datasets
    assert os.path.exists(dataset_paths), "File containing dataset paths does not exist."
    with open(dataset_paths,'r') as dpf:
        paths = dpf.read().splitlines()

    datasets = list()
    for fp in paths:
        assert os.path.exists(fp), "Dataset @ {} does not exist.".format(fp)

        try:
            # there is an internal validation of dataset
            ds = MLDataset(fp)
        except:
            print("Dataset @ {} is not a valid MLDataset!".format(fp))
            raise

        # add the valid dataset to list
        datasets.append(ds)

    # ensure same number of subjects across all datasets
    num_datasets = len(datasets)
    # looking into the first dataset
    common_ds = datasets[0]
    class_set, label_set, class_sizes = common_ds.summarize_classes()
    num_samples = common_ds.num_samples
    num_classes = len(class_set)

    for idx in range(1, num_datasets):
        this_ds = datasets[idx]
        assert num_samples==this_ds.num_samples, "Number of samples in different datasets differ!"
        assert class_set==set(this_ds.classes.values()), \
            "Classes differ among datasets! \n One dataset: {} \n Another: {}".format(
                class_set, set(this_ds.classes.values()))

    # re-map the labels (from 1 to n) to ensure numeric labels do not differ
    class_labels = list()
    for idx, cls in enumerate(class_set):
        class_labels[cls] = idx

    labels_with_correspondence = dict()
    for subid in common_ds.sample_ids:
        labels_with_correspondence[subid] = class_labels[common_ds.classes[subid]]

    for idx in range(num_datasets):
        datasets[idx].labels = labels_with_correspondence

    assert (train_perc >= 0.01 and train_perc <= 0.99), \
        "Training percentage {} out of bounds - must be > 0.01 and < 0.99".format(train_perc)

    num_features = np.zeros(num_datasets)
    for idx in range(num_datasets):
        num_features[idx] = datasets[idx].num_features

    # determine the common size for training
    print("Different classes in the training set are stratified to match the smallest class!")
    train_size_per_class = np.floor(train_perc*class_sizes).astype(np.float64)
    train_size_common = np.minimum(min(train_size_per_class), train_size_per_class)

    total_test_samples = np.sum(class_sizes) - np.sum(train_size_common)

    pred_prob_per_class    = np.full([num_repetitions, num_datasets, total_test_samples, num_classes], np.nan)
    pred_labels_per_rep_fs = np.full([num_repetitions, num_datasets, total_test_samples, 1], np.nan)
    test_labels_per_rep    = np.full([num_repetitions, total_test_samples], np.nan)

    best_min_leaf_size  = np.full([num_repetitions, num_datasets], np.nan)
    best_num_predictors = np.full([num_repetitions, num_datasets], np.nan)

    # initialize various lists
    misclf_sample_ids     = common_ds.sample_ids
    misclf_sample_classes = common_ds.classes

    num_times_tested  = np.zeros([num_samples, num_datasets])
    num_times_misclfd = np.zeros([num_samples, num_datasets])

    confusion_matrix  = np.full([num_classes, num_classes, num_repetitions, num_datasets], np.nan)
    accuracy_balanced = np.full([num_repetitions, num_datasets], np.nan)

    feature_importances_rf = [None]*num_datasets
    for idx in range(num_datasets):
        feature_importances_rf[idx] = np.full([num_repetitions,num_features[idx]], np.nan)

    # repeated-hold out CV begins here
    for rep in range(num_repetitions):
        # construct training and test sets (of sample ids)
        train_set = list()
        test_set = list()
        train_labels = list()
        test_labels = list()
        for idx, cls in enumerate(class_set):
            ids_in_class = common_ds.sample_ids_in_class(cls)

            # randomizing the list before selection
            # this is necessary to ensure resampling
            random.shuffle(ids_in_class)

            subset_train = ids_in_class[0:train_size_common]
            subset_test  = ids_in_class[train_size_common:]

            train_set.extend(subset_train)
            test_set.extend(subset_test)
            train_labels.extend([ common_ds.labels[sid] for sid in train_set ])
            test_labels.extend( [ common_ds.labels[sid] for sid in test_set  ])

        test_labels_per_rep[rep, :] = test_labels

        # evaluating each feature/dataset
        for dd in range(num_datasets):
            print(" Rep {:3d}, feature {:3d}".format(rep, dd))

            train_fs = datasets[dd].get_subset(train_set)
            test_fs  = datasets[dd].get_subset(test_set)

            pred_prob_per_class[rep, dd, :, :], pred_labels_per_rep_fs[rep, dd, :, :], \
                confmat, feature_importances_rf[rep, :], \
                best_min_leaf_size[rep, dd], best_num_predictors[rep, dd] = \
                clf_tuner(train_fs, test_fs)

            accuracy_balanced[rep,dd] = calc_balanced_accuracy(confmat)
            confusion_matrix[:,:,rep,dd] = confmat


    return


def ignore():
    "Empty method to ignore."

    pass


if __name__ == '__main__':
    ignore()