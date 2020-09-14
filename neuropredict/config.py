# import matplotlib.pyplot as plt
import numpy as np

# following affects the maximum num of predictors to be tried in random forest
PERC_PROB_ERROR_ALLOWED = 0.05

default_reduced_dim_size = 'tenth'
default_num_features_to_select = default_reduced_dim_size
feature_selection_size_methods = ('tenth', 'sqrt', 'log2', 'all')

variance_threshold = 0.001

# # ------- feature importance -------

violin_width = 0.6
violin_bandwidth = 0.15

CMAP_FEAT_IMP = 'hsv'
max_allowed_num_features_importance_map = 10

# Tree like estimators in sklearn return 0 for features that were never selected
# for training.
importance_value_to_treated_as_not_selected = 0.0
# importance values are sorted by value (higher better), so we need to able
# discard them
importance_value_never_tested = -np.Inf
importance_value_never_tested_stdev = np.nan

# to help bring up feature importances that can be very small in 4/5th decimal places
large_constant_factor = 1e5

# # ------- missing data imputation strategy -------------

missing_value_identifier = np.NaN  #
default_imputation_strategy = 'raise'
# not supporting 'constant' for now, as it is not popular,
#   and integrating it requires a bit more software engineering
avail_imputation_strategies = ('median', 'mean', 'most_frequent')
avail_imputation_strategies_with_raise = avail_imputation_strategies + \
                                         (default_imputation_strategy,)

missing_data_flag_name = 'missing_data'

## -------- preprocessing ----------

default_preprocessing_method = 'robustscaler'

## -------- covariates -------------

default_covariates = None
default_deconfounding_method = 'residualize'
avail_deconfounding_methods = ('residualize', 'augment')

# # ------- feature importance -------

importance_attr = {'randomforestclassifier'   : 'feature_importances_',
                   'extratreesclassifier'     : 'feature_importances_',
                   'decisiontreeclassifier'   : 'feature_importances_',
                   'svm'                      : 'coef_',
                   'xgboost'                  : 'feature_importances_',
                   'randomforestregressor'    : 'feature_importances_',
                   'extratreesregressor'      : 'feature_importances_',
                   'decisiontreeregressor'    : 'feature_importances_',
                   'GradientBoostingRegressor': 'feature_importances_',
                   'XGBoostRegressor'         : 'feature_importances_',
                   }

feat_imp_name = 'feat_importance'

# # ------- classifier

__classifier_CHOICES = ('RandomForestClassifier',
                        'ExtraTreesClassifier',
                        'DecisionTreeClassifier',
                        'SVM',
                        'XGBoost')
classifier_choices = [clf.lower() for clf in __classifier_CHOICES]

__regressor_CHOICES = ('RandomForestRegressor',
                       'ExtraTreesRegressor',
                       'DecisionTreeRegressor',
                       'SVR',
                       'KernelRidge',
                       'BayesianRidge',
                       'GaussianProcessRegressor',
                       'GradientBoostingRegressor',
                       'XGBoostRegressor'
                       )
regressor_choices = [clf.lower() for clf in __regressor_CHOICES]

__feat_sel_CHOICES = ('SelectKBest_mutual_info_classif',
                      'SelectKBest_f_classif',
                      'VarianceThreshold',
                      )
__dim_red_CHOICES = ('Isomap', 'LLE', 'LLE_modified', 'LLE_Hessian', 'LLE_LTSA')
__generic_fs_dr_CHOICES = __feat_sel_CHOICES + __dim_red_CHOICES
all_dim_red_methods = [fsm.lower() for fsm in __generic_fs_dr_CHOICES]

default_classifier = 'RandomForestClassifier'
default_regressor = 'RandomForestRegressor'

default_dim_red_method = 'VarianceThreshold'

__estimators_with_feature_importance = ('DecisionTreeClassifier',
                                        'RandomForestClassifier',
                                        'ExtraTreesClassifier',
                                        'LinearSVM',
                                        'XGBoost',
                                        'RandomForestRegressor',
                                        'ExtraTreesRegressor',
                                        'DecisionTreeRegressor',
                                        'XGBoostRegressor',
                                        )
estimators_with_feat_imp = [clf.lower() for clf in
                            __estimators_with_feature_importance]

additional_modules_reqd = {'xgboost': 'xgboost'}

# defines quantile_range parameter for RobustScaler
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
robust_scaler_iqr = (5, 95)

# Parameters specific to random forest classifier optimization
NUM_TREES = 250
NUM_TREES_RANGE = [250, 500]
NUM_TREES_STEP = 100
NUM_PREDICTORS_STEP = 2

MAX_MIN_LEAFSIZE = 5
LEAF_SIZE_STEP = 2

#### new workflow

workflow_types = ('classify', 'regress')
results_file_name = 'results_neuropredict.pkl'
options_file_name = 'options_neuropredict.pkl'
best_params_file_name = 'best_params_neuropredict.pkl'

prefix_dump = 'cv_results_quick_dump'

results_to_save = ['_workflow_type', '_checkpointing',
                   '_id_list', '_num_samples',
                   '_scoring', '_positive_class', '_positive_class_index',
                   '_target_set', '_train_set_size',
                   '_target_sizes', '_chance_accuracy',
                   'covariates', 'deconfounder',
                   'dim_red_method', 'grid_search_level', 'impute_strategy',
                   'num_procs', 'num_rep_cv', 'out_dir', 'pred_model',
                   'reduced_dim', 'results', 'train_perc',
                   'user_options']

### ---------  CV results class  ------------------------------------

_common_variable_set_to_load = ['_dataset_ids',
                               'attr', 'meta',
                               'metric_set', 'metric_val',
                               'num_rep', '_count',
                               'predicted_targets', 'true_targets', ]

clf_results_class_variables_to_load = _common_variable_set_to_load + \
                                      ['confusion_mat', 'misclfd_samplets', ]

regr_results_class_variables_to_load = _common_variable_set_to_load + ['residuals', ]

### ------------------------------------------------------------------------------

# CV
default_num_repetitions = 200
default_train_perc = 0.5

# model optimization
INNER_CV_NUM_SPLITS = 10
INNER_CV_TRAIN_PERC = 0.8
INNER_CV_TEST_PERC = 0.2

# when k-fold is chosen to be inner cv
INNER_CV_NUM_FOLDS = 5
INNER_CV_NUM_REPEATS = 5

# although refit is True by default in GridSearchCV, this is to avoid depending on
# defaults for future releases!
refit_best_model_on_ALL_training_set = True

# parallelization is now achieved at the repetitions level.
DEFAULT_NUM_PROCS = 4
GRIDSEARCH_PRE_DISPATCH = 1
GRIDSEARCH_NUM_JOBS = 1

GRIDSEARCH_LEVELS = ('none', 'light', 'exhaustive')
GRIDSEARCH_LEVEL_DEFAULT = GRIDSEARCH_LEVELS[0]

SEED_RANDOM = 652

PRECISION_METRICS = 2

## workflow

default_checkpointing = True

### performance metrics

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             r2_score, mean_absolute_error, explained_variance_score,
                             mean_squared_error)

default_scoring_metric = 'accuracy'

# allowed names: sorted(sklearn.metrics.SCORERS.keys())

default_metric_set_classification = (accuracy_score,
                                     balanced_accuracy_score)
default_metric_set_regression = (r2_score,
                                 mean_absolute_error,
                                 explained_variance_score,
                                 mean_squared_error)

alpha_regression_targets = 0.6
num_bins_hist = 50

# misclassifications
MISCLF_HIST_NUM_BINS = 20
MISCLF_PERC_THRESH = 0.6
MISCLF_HIST_ALPHA = 0.7
MISCLF_HIST_ANNOT_LINEWIDTH = 2

COMMON_FIG_SIZE = [9, 9]
LARGE_FIG_SIZE = [15, 15]
FONT_SIZE = 12
LINE_WIDTH = 1.5

FONT_SIZE_LARGE = 25
LINE_WIDTH_LARGE = 5

CMAP_DATASETS = 'Dark2'
CMAP_CONFMATX = 'cividis'  # 'viridis' # 'winter' # 'RdYlGn' # 'Blues' # plt.cm.Blues

file_name_options = 'options_neuropredict.pkl'
file_name_best_param_values = 'best_parameter_values.pkl'

quick_dump_prefix = 'cv_results_quick_dump'

max_len_identifiers = 75

output_dir_default = 'neuropredict_results'
temp_results_dir = 'temp_scratch_neuropredict'
temp_prefix_rhst = 'trial'
EXPORT_DIR_NAME = 'exported_results'
DELIMITER = ','
EXPORT_FORMAT = '%10.5f'

INPUT_FILE_FORMATS = ['.npy', '.numpy', '.csv', '.txt']

# API related defaults
default_feature_type = 'list_of_pyradigm_paths'

# when more than one feature set is given, which one to map everyone to
COMMON_DATASET_INDEX = 0

rhst_data_variables_to_persist = ['dataset_paths', 'method_names', 'train_perc',
                                  'num_repetitions', 'num_classes',
                                  'pred_prob_per_class', 'pred_labels_per_rep_fs',
                                  'test_labels_per_rep', 'best_params',
                                  'feature_importances_rf', 'feature_names',
                                  'num_times_misclfd', 'num_times_tested',
                                  'confusion_matrix', 'class_set', 'target_sizes',
                                  'accuracy_balanced', 'auc_weighted',
                                  'positive_class', 'classifier_name',
                                  'feat_select_method']

# TODO decide to where to include eTIV
# 'eTIV' is not included as it is used to norm subcortical volumes
freesurfer_whole_brain_stats_to_select = ['BrainSegVol', 'BrainSegVolNotVent',
                                          'lhCortexVol', 'rhCortexVol',
                                          'lhCorticalWhiteMatterVol',
                                          'rhCorticalWhiteMatterVol',
                                          'SubCortGrayVol', 'TotalGrayVol',
                                          'SupraTentorialVol',
                                          'SupraTentorialVolNotVent', 'MaskVol',
                                          'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV',
                                          'lhSurfaceHoles', 'rhSurfaceHoles', 'eTIV']

freesurfer_whole_brain_stats_to_ignore = ['SurfaceHoles',
                                          'CortexVol',
                                          'SupraTentorialVolNotVentVox',
                                          'CorticalWhiteMatterVol',
                                          'BrainSegVolNotVentSurf']

freesurfer_subcortical_seg_names_to_ignore = ['WM-hypointensities',
                                              'Left-WM-hypointensities',
                                              'Right-WM-hypointensities',
                                              'non-WM-hypointensities',
                                              'Left-non-WM-hypointensities',
                                              'Right-non-WM-hypointensities',
                                              'Optic-Chiasm']

freesurfer_subcortical_seg_names = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
                                    'Left-Cerebellum-White-Matter',
                                    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper',
                                    'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                                    '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
                                    'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                                    'Left-Accumbens-area', 'Left-VentralDC',
                                    'Left-vessel', 'Left-choroid-plexus',
                                    'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
                                    'Right-Cerebellum-White-Matter',
                                    'Right-Cerebellum-Cortex',
                                    'Right-Thalamus-Proper', 'Right-Caudate',
                                    'Right-Putamen', 'Right-Pallidum',
                                    'Right-Hippocampus', 'Right-Amygdala',
                                    'Right-Accumbens-area', 'Right-VentralDC',
                                    'Right-vessel', 'Right-choroid-plexus',
                                    '5th-Ventricle', 'CC_Posterior',
                                    'CC_Mid_Posterior', 'CC_Central',
                                    'CC_Mid_Anterior', 'CC_Anterior']
