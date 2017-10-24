# import matplotlib.pyplot as plt


# following affects the maximum num of predictors to be tried in random forest
PERC_PROB_ERROR_ALLOWED = 0.05

default_num_features_to_select = 'tenth'
feature_selection_size_methods = ('tenth', 'sqrt', 'log2', 'all')

variance_threshold = 0.001

# defines quantile_range parameter for RobustScaler
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
robust_scaler_iqr = (5, 95)

#Parameters specific to random forest classifier optimization
NUM_TREES = 250
NUM_TREES_RANGE = [250, 500]
NUM_TREES_STEP = 100
NUM_PREDICTORS_STEP = 2

MAX_MIN_LEAFSIZE = 5
LEAF_SIZE_STEP = 2

# model optimization
INNER_CV_NUM_SPLITS = 10
INNER_CV_TRAIN_PERC = 0.8
INNER_CV_TEST_PERC = 0.2

# parallelization is now achieved at the repetitions level.
DEFAULT_NUM_PROCS = 4
GRIDSEARCH_PRE_DISPATCH = 1
GRIDSEARCH_NUM_JOBS = 1

GRIDSEARCH_LEVELS = ('none', 'light', 'exhaustive')
GRIDSEARCH_LEVEL_DEFAULT = GRIDSEARCH_LEVELS[0]

SEED_RANDOM = 652

PRECISION_METRICS = 2

# misclassifications
MISCLF_HIST_NUM_BINS = 20
MISCLF_PERC_THRESH = 0.6
MISCLF_HIST_ALPHA = 0.7
MISCLF_HIST_ANNOT_LINEWIDTH = 2

COMMON_FIG_SIZE = [9, 9]
CMAP_DATASETS = 'Paired'
CMAP_CONFMATX = 'Blues' # plt.cm.Blues


CMAP_FEAT_IMP = 'hsv'
max_allowed_num_features_importance_map = 50

file_name_results = 'rhst_results.pkl'
EXPORT_DIR_NAME = 'exported_results'
DELIMITER = ','
EXPORT_FORMAT = '%10.5f'

INPUT_FILE_FORMATS = ['.npy', '.numpy', '.csv', '.txt']

# when more than one feature set is given, which one to map everyone to
COMMON_DATASET_INDEX = 0

rhst_data_variables_to_persist = ['dataset_paths', 'method_names', 'train_perc', 'num_repetitions', 'num_classes',
                  'pred_prob_per_class', 'pred_labels_per_rep_fs', 'test_labels_per_rep',
                  'best_params',
                  'feature_importances_rf', 'feature_names',
                  'num_times_misclfd', 'num_times_tested',
                  'confusion_matrix', 'class_set', 'class_sizes',
                  'accuracy_balanced', 'auc_weighted', 'positive_class']

# TODO decide to where to include eTIV
# 'eTIV' is not included as it is used to norm subcortical volumes
freesurfer_whole_brain_stats_to_select = [ 'BrainSegVol', 'BrainSegVolNotVent',
        'lhCortexVol', 'rhCortexVol',
        'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
        'SubCortGrayVol', 'TotalGrayVol',
        'SupraTentorialVol', 'SupraTentorialVolNotVent',
        'MaskVol', 'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV',
        'lhSurfaceHoles', 'rhSurfaceHoles',
        'eTIV' ]

freesurfer_whole_brain_stats_to_ignore = [ 'SurfaceHoles',
                                           'CortexVol',
                                           'SupraTentorialVolNotVentVox',
                                           'CorticalWhiteMatterVol',
                                           'BrainSegVolNotVentSurf']


freesurfer_subcortical_seg_names_to_ignore = ['WM-hypointensities', 'Left-WM-hypointensities', 'Right-WM-hypointensities',
                    'non-WM-hypointensities', 'Left-non-WM-hypointensities', 'Right-non-WM-hypointensities',
                    'Optic-Chiasm']

freesurfer_subcortical_seg_names = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                                    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
                                    'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus',
                                    'Left-Amygdala', 'CSF', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
                                    'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
                                    'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper',
                                    'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                                    'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
                                    'Right-choroid-plexus', '5th-Ventricle',
                                    'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']