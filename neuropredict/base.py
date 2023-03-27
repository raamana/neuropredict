from __future__ import print_function

import argparse
import pickle
import random
import sys
import textwrap
from abc import abstractmethod
from multiprocessing import Pool
from os import getcwd, makedirs
from os.path import abspath, exists as pexists, getsize, join as pjoin, realpath
from warnings import catch_warnings, filterwarnings, simplefilter

import numpy as np
from pyradigm.multiple import BaseMultiDataset
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from neuropredict import __version__, config as cfg
from neuropredict.algorithms import (compute_reduced_dimensionality, encode,
                                     get_deconfounder, get_preprocessor,
                                     make_pipeline)
from neuropredict.io import (get_metadata, get_metadata_in_pyradigm)
from neuropredict.results import ClassifyCVResults, RegressCVResults
from neuropredict.utils import (chance_accuracy, check_covariate_options,
                                check_num_procs, check_paths, impute_missing_data,
                                not_unspecified,
                                print_options, validate_feature_selection_size,
                                validate_impute_strategy)


class BaseWorkflow(object):
    """Class defining a structure for the neuropredict workflow"""


    def __init__(self,
                 datasets,
                 pred_model=cfg.default_classifier,
                 impute_strategy=cfg.default_imputation_strategy,
                 covariates=None,
                 deconfounder=cfg.default_deconfounding_method,
                 dim_red_method=cfg.default_dim_red_method,
                 reduced_dim=cfg.default_num_features_to_select,
                 train_perc=cfg.default_train_perc,
                 num_rep_cv=cfg.default_num_repetitions,
                 scoring=cfg.default_scoring_metric,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                 out_dir=None,
                 num_procs=cfg.DEFAULT_NUM_PROCS,
                 user_options=None,
                 checkpointing=False,
                 workflow_type='classify'
                 ):
        """Constructor"""

        self.datasets = datasets
        self.pred_model = pred_model
        self.impute_strategy = impute_strategy

        self.covariates = covariates
        self.deconfounder = deconfounder

        self.dim_red_method = dim_red_method
        self.reduced_dim = reduced_dim

        self.train_perc = train_perc
        self.num_rep_cv = num_rep_cv
        self._scoring = scoring
        self.grid_search_level = grid_search_level

        if out_dir is None:
            out_dir = getcwd()
        self.out_dir = out_dir
        self._fig_out_dir = pjoin(self.out_dir, 'figures')
        self._tmp_dump_dir = pjoin(self.out_dir, 'temp_dump')
        makedirs(self.out_dir, exist_ok=True)
        makedirs(self._fig_out_dir, exist_ok=True)
        makedirs(self._tmp_dump_dir, exist_ok=True)

        self._parall_proc = False
        self.num_procs = num_procs
        self.user_options = user_options
        self._checkpointing = checkpointing
        workflow_type = workflow_type.lower()
        if workflow_type not in cfg.workflow_types:
            raise ValueError('Invalid workflow. Must be one of {}'
                             ''.format(cfg.workflow_types))
        self._workflow_type = workflow_type

        if self.datasets is None:
            dataset_ids = 'noname_datasets'
        elif isinstance(self.datasets, BaseMultiDataset):
            dataset_ids = self.datasets.modality_ids
        else:
            dataset_ids = [f'dataset{ii}' for ii in range(len(self.datasets))]

        if self._workflow_type == 'classify':
            self.results = ClassifyCVResults(self._scoring, self.num_rep_cv,
                                             dataset_ids)
        else:
            self.results = RegressCVResults(self._scoring, self.num_rep_cv,
                                            dataset_ids)


    def _prepare(self):
        """Checks in inputs, parameters and their combinations"""

        self.num_rep_cv = int(self.num_rep_cv)
        if not np.isfinite(self.num_rep_cv):
            raise ValueError("Infinite number of repetitions is not recommended!")

        if self.num_rep_cv <= 1:
            raise ValueError("More than 1 repetition is necessary!")

        if self.train_perc <= 0.0 or self.train_perc >= 1.0:
            raise ValueError('Train perc > 0.0 and < 1.0')

        self.num_procs = check_num_procs(self.num_procs)

        if self.grid_search_level.lower() not in cfg.GRIDSEARCH_LEVELS:
            raise ValueError('Unrecognized level of grid search.'
                             ' Valid choices: {}'.format(cfg.GRIDSEARCH_LEVELS))

        # TODO for API use, pred_model and dim_reducer must be validated here again
        # if not isinstance(self.pred_model, BaseEstimator):

        self._id_list = list(self.datasets.samplet_ids)
        self._num_samples = len(self._id_list)
        self._train_set_size = np.int64(np.floor(self._num_samples * self.train_perc))
        self._train_set_size = max(1, min(self._num_samples, self._train_set_size))

        self._out_results_path = pjoin(self.out_dir, cfg.results_file_name)

        self._summarize_expt()


    def _summarize_expt(self):
        """Summarize the experiment for user info"""

        print('\nCURRENT EXPERIMENT:\n{line}'.format(line='-' * 50))
        print('Training percentage      : {:.2}'.format(self.train_perc))
        print('Number of CV repetitions : {}'.format(self.num_rep_cv))
        print('Number of processors     : {}'.format(self.num_procs))
        print('Dim reduction method     : {}'.format(self.dim_red_method))
        print('Dim reduction size       : {}'.format(self.reduced_dim))
        print('Predictive model chosen  : {}'.format(self.pred_model))
        print('Grid search level        : {}\n'.format(self.grid_search_level))

        if len(self.covariates) > 0:
            print('Covarites selected       : {}'.format(', '.join(self.covariates)))
            print('Deconfoudning method     : {}\n'.format(self.deconfounder))

        if self._workflow_type == 'classify':
            self._target_sizes = list(self.datasets.target_sizes.values())
            self._chance_accuracy = chance_accuracy(self._target_sizes, 'balanced')
            print('Estimated chance accuracy : {:.3f}\n'
                  ''.format(self._chance_accuracy))


    def run(self):
        """Full run of workflow"""

        self._prepare()

        if pexists(self._out_results_path) and getsize(self._out_results_path) > 0:
            print('Loading results from:\n {}\n'.format(self.out_dir))
            self.load()
        else:
            print('Saving results to: \n {}\n'.format(self.out_dir))

            # ignoring some not-so-critical warnings
            with catch_warnings():
                filterwarnings(action='once', category=UserWarning, module='joblib',
                               message='Multiprocessing-backed parallel loops '
                                       'cannot be nested, setting n_jobs=1')
                filterwarnings(action='once', category=UserWarning,
                               message='Some inputs do not have OOB scores')
                filterwarnings(action='once', category=UserWarning,
                               message='UserWarning: Features * are constant')
                np.seterr(divide='ignore', invalid='ignore')
                filterwarnings(action='once', category=RuntimeWarning,
                               message='invalid value encountered in true_divide')
                simplefilter(action='once', category=DeprecationWarning)
                # actual CV
                self._run_cv()

            self.save()

        self.summarize()
        self.visualize()

        print('Results have been saved to \n\t{}'.format(self._out_results_path))

        return self._out_results_path


    def _run_cv(self):
        """Actual CV"""

        if self.num_procs > 1:
            self._parall_proc = True
            self._checkpointing = True

            print('Parallelizing the repetitions of CV with {} processes ...'
                  ''.format(self.num_procs))
            with Pool(processes=self.num_procs) as pool:
                pool.map(self._single_run_cv, list(range(self.num_rep_cv)))
        else:
            # switching to regular sequential for loop to avoid any parallel drama
            for rep in range(self.num_rep_cv):
                self._single_run_cv(rep)


    def _single_run_cv(self, run_id=None):
        """Implements a single run of train, optimize and predict"""

        random.shuffle(self._id_list)
        train_set = self._id_list[:self._train_set_size]
        test_set = list(set(self._id_list) - set(train_set))

        for ds_id, ((train_data, train_targets), (test_data, test_targets)) \
                in self.datasets.get_subsets((train_set, test_set)):
            # print('Dataset {}'.format(ds_id))

            missing = self.datasets.get_attr(ds_id, cfg.missing_data_flag_name)
            if missing:
                train_data, test_data = impute_missing_data(
                        train_data, train_targets, self.impute_strategy, test_data)

            train_data, test_data = self._preprocess_data(train_data, test_data)

            # covariate regression / deconfounding WITHOUT using target values
            if len(self.covariates) > 0:
                train_covar, test_covar = self._get_covariates(train_set, test_set)
                train_data, test_data = self._deconfound_data(train_data, train_covar,
                                                              test_data, test_covar)
            # deconfounding targets could be added here in the future if needed

            best_pipeline, best_params, feat_importance = \
                self._optimize_pipeline_on_train_set(train_data, train_targets)

            self.results.add_attr(run_id, ds_id, 'feat_importance', feat_importance)

            self._eval_predictions(best_pipeline, test_data, test_targets,
                                   run_id, ds_id)

        # dump results if checkpointing is requested
        if self._checkpointing or self._parall_proc:
            self.results.dump(self._tmp_dump_dir, run_id)


    def _get_covariates(self, train_set, test_set):
        """Method to gather and organize covariate data"""

        train_covar, train_covar_dtypes = \
            self.datasets.get_common_attr(self.covariates, train_set)
        test_covar, test_covar_dtypes = \
            self.datasets.get_common_attr(self.covariates, test_set)

        if test_covar_dtypes != train_covar_dtypes:
            raise TypeError('covariate dtypes differ between train and test sets!\n'
                            ' Train: {}, Test: {}'
                            ''.format(train_covar_dtypes, test_covar_dtypes))

        train_covar, test_covar, _ = encode(train_covar, test_covar,
                                                   train_covar_dtypes)

        # column_stack ensures output is a 2D array, needed for sklearn transformers
        #   np.asarray() is needed to convert deprecated np.matrix to ndarray
        return \
            np.asarray(np.column_stack(train_covar)), \
            np.asarray(np.column_stack(test_covar))


    def _optimize_pipeline_on_train_set(self, train_data, train_targets):
        """Optimize model on training set."""

        reduced_dim = compute_reduced_dimensionality(
                self.reduced_dim, self._train_set_size, train_data.shape[1])
        pipeline, param_grid = make_pipeline(pred_model=self.pred_model,
                                             dim_red_method=self.dim_red_method,
                                             reduced_dim=reduced_dim,
                                             train_set_size=self._train_set_size,
                                             gs_level=self.grid_search_level)

        best_pipeline, best_params = self._optimize_pipeline(
                pipeline, train_data, train_targets, param_grid, self.train_perc)

        feat_importance = self._get_feature_importance(
                self.pred_model, best_pipeline, train_data.shape[1])

        return best_pipeline, best_params, feat_importance


    @staticmethod
    def _preprocess_data(train_data, test_data,
                         preproc_name=cfg.default_preprocessing_method):
        """Separate independent preprocessing of data"""

        preproc, preproc_name, preproc_param_grid = get_preprocessor(preproc_name)
        preproc.fit(train_data)
        return preproc.transform(train_data), preproc.transform(test_data)


    def _deconfound_data(self, train_data, train_covar, test_data, test_covar):
        """Builds custom composite deconfounder holding both train and test covar"""

        deconf, _, deconf_param_grid = get_deconfounder(
                self.deconfounder, self.grid_search_level)

        # training based solely on training set only, both features and confounds
        # TODO hyper-parameter optim not done for deconfounding methods
        deconf.fit(train_data, train_covar)

        # returning the deconfounded data
        return deconf.transform(train_data, train_covar), \
               deconf.transform(test_data, test_covar)


    @staticmethod
    def _get_feature_importance(est_name, pipeline, num_features, fill_value=np.nan):
        """Extracts the feature importance of input features, if available."""

        if est_name not in cfg.importance_attr:
            # some estimators simply do not provide it
            feat_importance = None
        else:
            # assuming order in pipeline construction :
            #   - step 0 : feature selector / dim reducer
            #   - step 1 : estimator
            # pipeline.steps returns a tuple (name, est_object),
            #   so [1] is necessary to access the actual object
            dim_red = pipeline.steps[0][1]
            est = pipeline.steps[-1][1]  # the final step in an sklearn pipeline
            #   is always an estimator/classifier

            feat_importance = None
            if hasattr(dim_red, 'get_support'):  # nonlinear dim red won't have this
                index_selected_features = dim_red.get_support(indices=True)

                if hasattr(est, cfg.importance_attr[est_name]):
                    feat_importance = np.full(num_features, fill_value)
                    feat_importance[index_selected_features] = \
                        getattr(est, cfg.importance_attr[est_name])

        return feat_importance


    @staticmethod
    def _optimize_pipeline(pipeline, train_data, train_targets,
                           param_grid, train_perc_inner_cv):
        """Optimizes a given pipeline on the given dataset"""

        # TODO perhaps k-fold is a better inner CV,
        #   which guarantees full use of training set with fewer repeats?
        inner_cv = ShuffleSplit(n_splits=cfg.INNER_CV_NUM_SPLITS,
                                train_size=train_perc_inner_cv,
                                test_size=1.0 - train_perc_inner_cv)
        # inner_cv = RepeatedKFold(n_splits=cfg.INNER_CV_NUM_FOLDS,
        #   n_repeats=cfg.INNER_CV_NUM_REPEATS)

        # gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv,
        #                   n_jobs=cfg.GRIDSEARCH_NUM_JOBS,
        #                   pre_dispatch=cfg.GRIDSEARCH_PRE_DISPATCH)

        # not specifying n_jobs to avoid any kind of parallelism (joblib) from within
        # sklearn to avoid potentially bad interactions with outer parallelization
        # with builtin multiprocessing library
        gs = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=inner_cv,  # TODO using default scoring metric?
                          refit=cfg.refit_best_model_on_ALL_training_set)
        gs.fit(train_data, train_targets)

        return gs.best_estimator_, gs.best_params_


    @abstractmethod
    def _eval_predictions(self, pipeline, test_data, true_targets, run_id, ds_id):
        """
        Evaluate predictions and perf estimates to results class.
        Prints a quick summary too, as an indication of progress.

        Making it abstract to let the child classes decide on what types of
        predictions to make (probabilistic or not), and how to evaluate them
        """


    def save(self):
        """Saves the results and state to disk."""

        if self._parall_proc:
            self.results.gather_dumps(self._tmp_dump_dir)

        out_dict = {var: getattr(self, var, None) for var in cfg.results_to_save}

        try:
            with open(self._out_results_path, 'wb') as res_fid:
                pickle.dump(out_dict, res_fid)
        except:
            raise IOError('Error saving the results to disk!\nOut path:{}'
                          ''.format(self._out_results_path))
        else:
            print('\nResults saved to {}\n'.format(self._out_results_path))
            # cleanup
            try:
                from shutil import rmtree
                rmtree(self._tmp_dump_dir, ignore_errors=True)
            except:
                print('Error in removing temp dir - remove it yourself:\n{}'
                      ''.format(self._tmp_dump_dir))

        return self._out_results_path


    def load(self, direct_path=None):
        """Mechanism to reload results.

        Useful for check-pointing, and restore upon crash etc
        """

        if direct_path is not None and pexists(direct_path):
            # this helps with propagating path to other parts of workflow,
            # such as self.visualize()
            self._out_results_path = direct_path

        try:
            with open(self._out_results_path, 'rb') as res_fid:
                loaded_results = pickle.load(res_fid)
        except:
            raise IOError('Error loading the results from path: {}'
                          ''.format(self._out_results_path))
        else:
            print('\nResults loaded from {}\n'.format(self._out_results_path))

        for var in cfg.results_to_save:
            setattr(self, var, loaded_results[var])

        # TODO need some simple validation to ensure loaded results are usable

        print()


    @abstractmethod
    def summarize(self):
        """Simple summary of the results produced, for logging and user info"""


    @abstractmethod
    def visualize(self):
        """Method to produce all the relevant visualizations based on the results
        from this workflow."""


    def redo_visualizations(self, direct_path):
        """Helper to [re-]produce the visualizations from existing results"""

        if pexists(direct_path) and getsize(direct_path) > 0:
            print('Loading results from:\n {}\n'.format(direct_path))
            self.load(direct_path)

        self.visualize()


    def _plot_feature_importance(self):
        """Bar plot comparing feature importance"""

        from neuropredict.visualize import feature_importance_map
        fig_base_out_path = pjoin(self._fig_out_dir, 'feature_importance')

        feat_imp = self.results.attr[cfg.feat_imp_name]

        # check if the all values are None or NaN
        unusable = list()
        for method_fi in feat_imp.values():
            if method_fi is None:
                unusable.append(True)
            else:
                unusable.append(np.all(np.isnan(method_fi.flatten())))
        if np.all(unusable):  # no feat imp usable/available
            print('\nFeature importance for this run are not available!\n')
            return

        out_feat_imp = list()
        feat_names = list()
        for idx, ds in enumerate(self.datasets.modality_ids):
            num_features = feat_imp[(ds, 0)].size
            fi_arr = np.empty((self.num_rep_cv, num_features))
            for run in range(self.num_rep_cv):
                fi_arr[run, :] = feat_imp[(ds, run)]
            out_feat_imp.append(fi_arr)
            feat_names.append(self.datasets.feature_names[ds])

        feature_importance_map(out_feat_imp, self.datasets.modality_ids,
                               fig_base_out_path, feature_names=feat_names)


def get_parser_base():
    """Parser to specify arguments and their defaults."""

    help_text_pyradigm_paths = textwrap.dedent("""
    Path(s) to pyradigm datasets.

    Each path is self-contained dataset identifying each sample, its class and 
    features.
    \n \n """)

    help_text_user_defined_folder = textwrap.dedent("""
        List of absolute paths to user's own features.

        Format: Each of these folders contains a separate folder for each subject 
        ( named after its ID in the metadata file) containing a file called 
        features.txt with one number per line. All the subjects (in a given 
        folder) must have the number of features ( #lines in file). Different 
        parent folders (describing one feature set) can have different number of 
        features for each subject, but they must all have the same number of 
        subjects (folders) within them.

        Names of each folder is used to annotate the results in visualizations. 
        Hence name them uniquely and meaningfully, keeping in mind these figures 
        will be included in your papers. For example,

        .. parsed-literal::

            --user_feature_paths /project/fmri/ /project/dti/ /project/t1_volumes/

        Only one of the ``--pyradigm_paths``, ``user_feature_paths``, 
        ``data_matrix_path`` or ``arff_paths`` options can be specified.
        \n \n """)

    help_text_data_matrix = textwrap.dedent("""
    List of absolute paths to text files containing one matrix of size N x p  (
    num_samples x num_features).

    Each row in the data matrix file must represent data corresponding to sample 
    in the same row of the meta data file (meta data file and data matrix must be 
    in row-wise correspondence).

    Name of this file will be used to annotate the results and visualizations.

    E.g. ``--data_matrix_paths /project/fmri.csv /project/dti.csv 
    /project/t1_volumes.csv``

    Only one of ``--pyradigm_paths``, ``user_feature_paths``, ``data_matrix_path`` 
    or ``arff_paths`` options can be specified.

    File format could be
    
     - a simple comma-separated text file (with extension .csv or .txt), which can 
     easily be read back with ``numpy.loadtxt(filepath, delimiter=',')``, or
     
     - a numpy array saved to disk (with extension .npy or .numpy) that can read 
     in with ``numpy.load(filepath)``.

     One could use ``numpy.savetxt(data_array, delimiter=',')`` or ``numpy.save(
     data_array)`` to save features.

     File format is inferred from its extension.
     \n \n """)

    help_text_train_perc = textwrap.dedent("""
    Percentage of the smallest class to be reserved for training.

    Must be in the interval [0.01 0.99].

    If sample size is sufficiently big, we recommend 0.5.
    If sample size is small, or class imbalance is high, choose 0.8.
    \n \n """)

    help_text_num_rep_cv = textwrap.dedent("""
    Number of repetitions of the repeated-holdout cross-validation.

    The larger the number, more stable the estimates will be.
    \n \n """)

    help_text_metadata_file = textwrap.dedent("""
    Abs path to file containing metadata for subjects to be included for analysis.

    At the minimum, each subject should have an id per row followed by the class 
    it belongs to.

    E.g.
    .. parsed-literal::

        sub001,control
        sub002,control
        sub003,disease
        sub004,disease

    \n \n """)

    help_text_dimensionality_red_size = textwrap.dedent("""
    Number of features to select as part of feature selection. Options:

         - 'tenth'
         - 'sqrt'
         - 'log2'
         - 'all'
         - or an integer ``k`` <= min(dimensionalities from all dataset)

    Default: ``{}`` of the number of samples in the training set.

    For example, if your dataset has 90 samples, you chose 50 percent for training 
    (default), then Y will have 90*.5=45 samples in training set, leading to 5 
    features to be selected for training. If you choose a fixed integer ``k``, 
    ensure all the feature sets under evaluation have atleast ``k`` features.
    \n \n """.format(cfg.default_num_features_to_select))

    help_text_gs_level = textwrap.dedent("""
    Flag to specify the level of grid search during hyper-parameter optimization 
    on the training set.
    
    Allowed options are : 'none', 'light' and 'exhaustive', in the order of how 
    many values/values will be optimized. More parameters and more values demand 
    more resources and much longer time for optimization.

    The 'light' option tries to "folk wisdom" to try least number of values (no 
    more than one or two), for the parameters for the given classifier. (e.g. a 
    lage number say 500 trees for a random forest optimization). The 'light' will 
    be the fastest and should give a "rough idea" of predictive performance. The 
    'exhaustive' option will try to most parameter values for the most parameters 
    that can be optimized.
    """)

    help_text_make_vis = textwrap.dedent("""
    Option to make visualizations from existing results in the given path. 
    This is helpful when neuropredict failed to generate result figures 
    automatically 
    e.g. on a HPC cluster, or another environment when DISPLAY is either not 
    available.

    """)

    help_text_num_cpus = textwrap.dedent("""
    Number of CPUs to use to parallelize CV repetitions.

    Default : 4.

    Number of CPUs will be capped at the number available on the machine if higher 
    is requested.
    \n \n """)

    help_text_out_dir = textwrap.dedent("""
    Output folder to store gathered features & results.
    \n \n """)

    help_dim_red_method = textwrap.dedent("""
    Feature selection, or dimensionality reduction method to apply prior to 
    training the classifier.

    **NOTE**: when feature 'selection' methods are used, we are able to keep track 
    of which features in the original input space were selected and hence visualize 
    their feature importance after the repetitions of CV. When the more generic 
    'dimensionality reduction' methods are used, *features often get transformed to 
    new subspaces*, wherein the link to original features is lost. Hence, 
    importance values for original input features can not be computed and,  
    are not visualized.

    Default: ``{}``, removing features with 0.001 percent of lowest 
    variance (zeros etc).

    """.format(cfg.default_dim_red_method))

    help_imputation_strategy = textwrap.dedent("""
    Strategy to impute any missing data (as encoded by NaNs).

    Default: 'raise', which raises an error if there is any missing data anywhere.
    Currently available imputation strategies are: {}

    """.format(cfg.avail_imputation_strategies))

    help_covariate_list = textwrap.dedent("""
    List of covariates to be taken into account. They must be present in the 
    original feature set in pyradigm format, which is required to implement the 
    deconfounding (covariate regression) properly. The pyradigm data structure 
    allows you to specify data type (categorical or numerical) for each 
    covariate/attribute, which is necessary to encode them accurately. 
    
    Specify them as a space-separated list of strings (with each name containing no
    spaces or any special characters), exactly as you encoded them in the input 
    pyradigm dataset. Example: ``-cl age site``

    """)

    help_covariate_method = textwrap.dedent("""
    Type of "deconfounding" method to handle confounds/covariates. This method 
    would be trained on the training set only (not their targets, just features 
    and covariates). The trained model is then used to deconfound both training 
    features (prior to fitting the predictive model) and  to transform the test 
    set prior to making predictions on them. 

    Available choices: {}
    
    """.format(cfg.avail_deconfounding_methods))

    help_text_print_options = textwrap.dedent("""
    Prints the options used in the run in an output folder.

    """)

    parser = argparse.ArgumentParser(prog="neuropredict",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='Easy, standardized and '
                                                 'comprehensive predictive '
                                                 'analysis.')

    parser.add_argument("-m", "--meta_file", action="store", dest="meta_file",
                        default=None, required=False, help=help_text_metadata_file)

    parser.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    user_feat_args = parser.add_argument_group(title='Input data and formats',
                                               description='Only one of the '
                                                           'following types can be '
                                                           'specified.')

    user_feat_args.add_argument("-y", "--pyradigm_paths", action="store",
                                dest="pyradigm_paths",
                                nargs='+',  # to allow for multiple features
                                default=None,
                                help=help_text_pyradigm_paths)

    user_feat_args.add_argument("-u", "--user_feature_paths", action="store",
                                dest="user_feature_paths",
                                nargs='+',  # to allow for multiple features
                                default=None,
                                help=help_text_user_defined_folder)

    user_feat_args.add_argument("-d", "--data_matrix_paths", action="store",
                                dest="data_matrix_paths",
                                nargs='+',
                                default=None,
                                help=help_text_data_matrix)

    cv_args = parser.add_argument_group(title='Cross-validation',
                                        description='Parameters related to '
                                                    'training and '
                                                    'optimization during '
                                                    'cross-validation')

    cv_args.add_argument("-t", "--train_perc", action="store",
                         dest="train_perc",
                         default=cfg.default_train_perc,
                         help=help_text_train_perc)

    cv_args.add_argument("-n", "--num_rep_cv", action="store",
                         dest="num_rep_cv",
                         default=cfg.default_num_repetitions,
                         help=help_text_num_rep_cv)

    cv_args.add_argument("-k", "--reduced_dim_size",
                         dest="reduced_dim_size",
                         action="store",
                         default=cfg.default_reduced_dim_size,
                         help=help_text_dimensionality_red_size)

    cv_args.add_argument("-g", "--gs_level", action="store", dest="gs_level",
                         default="light", help=help_text_gs_level,
                         choices=cfg.GRIDSEARCH_LEVELS, type=str.lower)

    pipeline_args = parser.add_argument_group(
            title='Predictive Model',
            description='Parameters of pipeline comprising the predictive model')

    pipeline_args.add_argument("-is", "--impute_strategy", action="store",
                               dest="impute_strategy",
                               default=cfg.default_imputation_strategy,
                               help=help_imputation_strategy,
                               choices=cfg.avail_imputation_strategies_with_raise,
                               type=str.lower)

    pipeline_args.add_argument("-cl", "--covariates", action="store",
                               dest="covariates",
                               nargs='+',
                               default=cfg.default_covariates,
                               help=help_covariate_list)

    pipeline_args.add_argument("-cm", "--covar_method", action="store",
                               dest="covar_method",
                               default=cfg.default_deconfounding_method,
                               help=help_covariate_method,
                               type=str.lower)

    pipeline_args.add_argument("-dr", "--dim_red_method", action="store",
                               dest="dim_red_method",
                               default=cfg.default_dim_red_method,
                               help=help_dim_red_method,
                               choices=cfg.all_dim_red_methods,
                               type=str.lower)

    vis_args = parser.add_argument_group(
            title='Visualization',
            description='Parameters related to generating visualizations')

    vis_args.add_argument("-z", "--make_vis", action="store", dest="make_vis",
                          default=None, help=help_text_make_vis)

    comp_args = parser.add_argument_group(
            title='Computing',
            description='Parameters related to computations/debugging')

    comp_args.add_argument("-c", "--num_procs", action="store", dest="num_procs",
                           default=cfg.DEFAULT_NUM_PROCS, help=help_text_num_cpus)

    comp_args.add_argument("--po", "--print_options", action="store",
                           dest="print_opt_dir",
                           default=False, help=help_text_print_options)

    comp_args.add_argument('-v', '--version', action='version',
                           version='%(prog)s {version}'.format(version=__version__))

    return parser, user_feat_args, cv_args, pipeline_args, vis_args, comp_args


def parse_common_args(parser):
    """Common utility to parse common CLI args"""

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        user_args = parser.parse_args()
    except:
        parser.exit(1)

    if len(sys.argv) == 3:
        # only if no features were specified to be assessed
        if not any(not_unspecified(getattr(user_args, attr))
                   for attr in ('user_feature_paths', 'data_matrix_paths',
                                'pyradigm_paths', 'arff_paths')):

            if not_unspecified(user_args.print_opt_dir) and user_args.print_opt_dir:
                run_dir = realpath(user_args.print_opt_dir)
                print_options(run_dir)

            if not_unspecified(user_args.make_vis):
                out_dir = realpath(user_args.make_vis)
                res_path = pjoin(out_dir, cfg.results_file_name)
                if pexists(out_dir) and pexists(res_path):
                    if not_unspecified(user_args.make_vis):
                        print('\n\nSaving the visualizations to \n{}'
                              ''.format(out_dir))
                        cli_prog_name = sys.argv[0].lower()
                        if cli_prog_name in ('np_classify', 'neuropredict_classify'):
                            from neuropredict.classify import ClassificationWorkflow
                            clf_expt = ClassificationWorkflow(datasets=None)
                            clf_expt.redo_visualizations(res_path)
                        elif cli_prog_name in ('np_regress', 'neuropredict_regress'):
                            from neuropredict.regress import RegressionWorkflow
                            reg_expt = RegressionWorkflow(datasets=None)
                            reg_expt.redo_visualizations(res_path)
                        else:
                            raise ValueError(
                                    'Incorrect CLI command invoked for --make_vis. '
                                    'It must be either np_classify or np_regress. '
                                    'Or their longer forms neuropredict_classify '
                                    'or neuropredict_regress.')

                else:
                    raise ValueError('Given folder does not exist, '
                                     'or has no results file!')

            sys.exit(0)

    user_feature_paths, user_feature_type, fs_subject_dir, meta_data_path, \
    meta_data_format = organize_inputs(user_args)

    if not meta_data_path:
        if user_args.meta_file is not None:
            meta_file = abspath(user_args.meta_file)
            if not pexists(meta_file):
                raise IOError("Meta data file doesn't exist.")
        else:
            raise ValueError('Metadata file must be provided '
                             'when not using pyradigm/ARFF inputs.')

        sample_ids, classes = get_metadata(meta_file)
    else:
        print('Using meta data from:\n\t{}\n'.format(meta_data_path))
        sample_ids, classes = get_metadata_in_pyradigm(meta_data_path,
                                                       meta_data_format)

    if user_args.out_dir is not None:
        out_dir = realpath(user_args.out_dir)
    else:
        out_dir = pjoin(realpath(getcwd()), cfg.output_dir_default)

    try:
        makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Output folder could not be created.')

    train_perc = np.float32(user_args.train_perc)
    if not (0.01 <= train_perc <= 0.99):
        raise ValueError("Training percentage {} out of bounds "
                         "- must be >= 0.01 and <= 0.99".format(train_perc))

    num_rep_cv = np.int64(user_args.num_rep_cv)
    if num_rep_cv < 10:
        raise ValueError("Atleast 10 repetitions of CV is recommened.")

    num_procs = check_num_procs(user_args.num_procs)

    reduced_dim_size = validate_feature_selection_size(
            user_args.reduced_dim_size)

    impute_strategy = validate_impute_strategy(user_args.impute_strategy)

    covar_list, covar_method = check_covariate_options(
            user_args.covariates, user_args.covar_method)

    grid_search_level = user_args.gs_level.lower()
    if grid_search_level not in cfg.GRIDSEARCH_LEVELS:
        raise ValueError('Unrecognized level of grid search. Valid choices: {}'
                         ''.format(cfg.GRIDSEARCH_LEVELS))

    dim_red_method = user_args.dim_red_method.lower()

    return user_args, user_feature_paths, user_feature_type, fs_subject_dir, \
           meta_data_path, meta_data_format, sample_ids, classes, out_dir, \
           train_perc, num_rep_cv, num_procs, reduced_dim_size, impute_strategy, \
           covar_list, covar_method, grid_search_level, dim_red_method


def organize_inputs(user_args):
    """
    Validates the input features specified and
    returns organized list of paths and readers.

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
    # specifying pyradigm avoids the need for separate meta data file
    meta_data_supplied = False
    meta_data_format = None

    if hasattr(user_args, 'fs_subject_dir') and \
            not_unspecified(user_args.fs_subject_dir):
        fs_subject_dir = abspath(user_args.fs_subject_dir)
        if not pexists(fs_subject_dir):
            raise IOError("Given Freesurfer directory doesn't exist.")
        atleast_one_feature_specified = True
    else:
        fs_subject_dir = None

    # ensuring only one type is specified
    mutually_excl_formats = ['user_feature_paths',
                             'data_matrix_paths',
                             'pyradigm_paths',
                             'arff_paths']
    not_none_count = 0
    for fmt in mutually_excl_formats:
        if hasattr(user_args, fmt) and \
                not_unspecified(getattr(user_args, fmt)):
            not_none_count = not_none_count + 1
    if not_none_count > 1:
        raise ValueError('Only one of the following formats can be specified:\n'
                         '{}'.format(mutually_excl_formats))

    if hasattr(user_args, 'user_feature_paths') and \
            not_unspecified(user_args.user_feature_paths):
        user_feature_paths = check_paths(user_args.user_feature_paths,
                                         path_type='user defined (dir_of_dirs)')
        atleast_one_feature_specified = True
        user_feature_type = 'dir_of_dirs'

    elif hasattr(user_args, 'data_matrix_paths') and \
            not_unspecified(user_args.data_matrix_paths):
        user_feature_paths = check_paths(user_args.data_matrix_paths,
                                         path_type='data matrix')
        atleast_one_feature_specified = True
        user_feature_type = 'data_matrix'

    elif hasattr(user_args, 'pyradigm_paths') and \
            not_unspecified(user_args.pyradigm_paths):
        user_feature_paths = check_paths(user_args.pyradigm_paths,
                                         path_type='pyradigm')
        atleast_one_feature_specified = True
        meta_data_supplied = user_feature_paths[0]
        meta_data_format = 'pyradigm'
        user_feature_type = 'pyradigm'

    elif hasattr(user_args, 'arff_paths') and \
            not_unspecified(user_args.arff_paths):
        user_feature_paths = check_paths(user_args.arff_paths, path_type='ARFF')
        atleast_one_feature_specified = True
        user_feature_type = 'arff'
        meta_data_supplied = user_feature_paths[0]
        meta_data_format = 'arff'
    else:
        user_feature_paths = None
        user_feature_type = None

    # map in python 3 returns a generator, not a list, so len() wouldn't work
    if not isinstance(user_feature_paths, list):
        user_feature_paths = list(user_feature_paths)

    if not atleast_one_feature_specified:
        raise ValueError('Atleast one method specifying features must be specified. '
                         'It can be a path(s) to pyradigm dataset, matrix file, '
                         'user-defined folder or a FreeSurfer subject directory.')

    return user_feature_paths, user_feature_type, fs_subject_dir, \
           meta_data_supplied, meta_data_format


class NeuroPredictException(Exception):
    """Custom exception to distinguish neuropredict related errors (usage etc)
    from the usual."""
    pass


class MissingDataException(NeuroPredictException):
    """Custom exception to uniquely identify this error. Helpful for testing etc"""
    pass
