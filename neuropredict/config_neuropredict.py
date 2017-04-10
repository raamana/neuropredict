import matplotlib.pyplot as plt

NUM_TREES = 100
NUM_PREDICTORS_STEP = 2
# following affects the maximum num of predictors to be tried in random forest
PERC_PROB_ERROR_ALLOWED = 0.05

MAX_MIN_LEAFSIZE = 20
LEAF_SIZE_STEP = 5

SEED_RANDOM = 652

# misclassifications
MISCLF_HIST_NUM_BINS = 20
MISCLF_PERC_THRESH = 0.6
MISCLF_HIST_ALPHA = 0.7
MISCLF_HIST_ANNOT_LINEWIDTH = 2

COMMON_FIG_SIZE = [9, 9]
CMAP_DATASETS = 'Paired'
CMAP_CONFMATX = plt.cm.Blues
CMAP_FEAT_IMP = 'hsv'

file_name_results = 'rhst_results.pkl'
EXPORT_DIR_NAME = 'exported_results'
DELIMITER = ','
EXPORT_FORMAT = '%10.5f'

INPUT_FILE_FORMATS = ['.npy', '.numpy', '.csv', '.txt']

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