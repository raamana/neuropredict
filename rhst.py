
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import pickle
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

file_name_results = "rhst_results.pkl"

from pyradigm import MLDataset


def eval_optimized_clsfr_on_testset(train_fs, test_fs):
    "Method to optimize the classifier on the training set and returns predictions on test set. "

    MAX_DIM_TRAINING = max_dimensionality_to_avoid_curseofdimensionality(train_fs.num_samples)

    NUM_TREES = 50
    MAX_MIN_LEAFSIZE = 20
    SEED_RANDOM = 652

    range_min_leafsize   = range(1, MAX_MIN_LEAFSIZE, 5)
    range_num_predictors = range(1, MAX_DIM_TRAINING, 2)
    oob_error_train = np.full([len(range_min_leafsize), len(range_num_predictors)], np.nan)

    for idx_ls, minls in enumerate(range_min_leafsize):
        for idx_np, num_pred in enumerate(range_num_predictors):
            rf = RandomForestClassifier(n_estimators=NUM_TREES, max_depth=None,
                                        max_features=num_pred , min_samples_leaf = minls,
                                        random_state=SEED_RANDOM)
            rf.fit(train_fs.data, train_fs.target)
            oob_error_train[idx_ls, idx_np] = rf.oob_score_

    # identifying the best parameters
    best_idx_ls, best_idx_numpred = np.unravel_index(oob_error_train.argmin(), oob_error_train.shape)
    best_minleafsize    = range_min_leafsize[best_idx_ls]
    best_num_predictors = range_num_predictors[best_idx_numpred]

    # training the RF using the best parameters
    best_rf = RandomForestClassifier( max_features=best_num_predictors, min_samples_leaf=best_minleafsize,
                                      n_estimators=NUM_TREES, random_state=SEED_RANDOM)

    # making predictions on the test set and assessing their performance
    pred_labels = best_rf.predict(test_fs.data)
    feat_importance = best_rf.feature_importances_

    #TODO test if the gathering of prob data is consistent across multiple calls to this method
    #   perhaps by controlling the class order in input
    # The order of the classes corresponds to that in the attribute best_rf.classes_.
    pred_prob = best_rf.predict_proba

    conf_mat = confusion_matrix(test_fs.target, pred_labels)

    return pred_prob, pred_labels, conf_mat, feat_importance, best_minleafsize, best_num_predictors


def display_confusion_matrix(cfmat, class_labels,
                             title='Confusion matrix',
                             display_perc=False,
                             cmap=plt.cm.Blues):
    """
    Display routine for the confusion matrix.
    Entries in confusin matrix can be turned into percentages with `display_perc=True`.
    """

    plt.imshow(cfmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    if display_perc:
        cfmat = cfmat.astype('float') / cfmat.sum(axis=1)[:, np.newaxis]

    # trick from sklearn
    thresh = cfmat.max() / 2.
    for i, j in itertools.product(range(cfmat.shape[0]), range(cfmat.shape[1])):
        plt.text(j, i, cfmat[i, j],
                 horizontalalignment="center",
                 color="white" if cfmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')


def max_dimensionality_to_avoid_curseofdimensionality(num_samples, perc_prob_error_allowed = 0.05):
    """
    Computes the largest dimensionality that can be used to train a predictive model
        avoiding curse of dimensionality for a 5% probability of error.
    Optional argument can specify the amount of error that the user wants to allow (deafult : 5% prob of error ).

    Citation: Michael Fitzpatrick and Milan Sonka, Handbook of Medical Imaging, 2000

    :param num_samples:
    :param perc_prob_error_allowed:
    :return:
    """

    if num_samples < 1/(2*perc_prob_error_allowed):
        max_red_dim = 1
    else:
        max_red_dim = np.floor(num_samples * (2*perc_prob_error_allowed))

    return max_red_dim


def balanced_accuracy(confmat):
    "Computes the balanced accuracy in a given confusion matrix!"
    # IMPLEMENT TODO

    pass

def run(dataset_path_file, out_results_dir, train_perc = 0.8, num_repetitions = 200):
    """

    :param dataset_path_file: path to file containing list of paths (each containing a valid MLDataset).
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


    assert os.path.exists(dataset_path_file), "File containing dataset paths does not exist."
    with open(dataset_path_file, 'r') as dpf:
        dataset_paths = dpf.read().splitlines()

    try:
        out_results_dir = os.path.abspath(out_results_dir)
        if os.path.exists(out_results_dir):
            os.mkdir(out_results_dir)
    except:
        raise IOError('Error in checking or creating output directiory. Ensure write permissions!')

    # loading datasets
    datasets = list()
    for fp in dataset_paths:
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

        # test/train sets are common across different featsets being tested
        test_labels_per_rep[rep, :] = test_labels

        # evaluating each feature/dataset
        for dd in range(num_datasets):
            print(" Rep {:3d}, feature {:3d}:".format(rep, dd))

            train_fs = datasets[dd].get_subset(train_set)
            test_fs  = datasets[dd].get_subset(test_set)

            pred_prob_per_class[rep, dd, :, :], pred_labels_per_rep_fs[rep, dd, :, :], \
                confmat, feature_importances_rf[rep, :], \
                best_min_leaf_size[rep, dd], best_num_predictors[rep, dd] = \
                eval_optimized_clsfr_on_testset(train_fs, test_fs)

            accuracy_balanced[rep,dd] = balanced_accuracy(confmat)
            confusion_matrix[:,:,rep,dd] = confmat

            # TODO tabulating misclassifications
            # misclfd_sample_ids = test_set



    # TODO generate visualizations for each feature set as well as a comparative summary!
    # TODO generate a CSV of different metrics for each dataset, as well as a reloadable

    # save results
    var_list_to_save = [dataset_paths, train_perc, num_repetitions, num_classes,
                        pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep,
                        best_min_leaf_size, best_num_predictors, feature_importances_rf,
                        misclf_sample_ids, misclf_sample_classes, num_times_misclfd, num_times_tested,
                        confusion_matrix, accuracy_balanced ]

    out_results_path = os.path.join(out_results_dir, file_name_results)
    with open(out_results_path, 'wb') as cfg:
        pickle.dump(var_list_to_save, cfg)

    return


if __name__ == '__main__':
    pass