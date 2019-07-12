__all__ = ['get_pipeline', 'get_feature_importance',]

from neuropredict import config_neuropredict as cfg
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest, VarianceThreshold
from sklearn.pipeline import Pipeline


def make_parameter_grid(estimator_name=None, named_ranges=None):
    """
        Builds a sklearn.pipeline compatible dict of named ranges,
        given an list of tuples, wherein each element is (param_name, param_values).
        Here, a param_name is the name of an estimator's parameter, and
            param_values are an iterable of values for that parameter.

    Parameters
    ----------
    estimator_name : str
        Valid python identifier to name the current step of the pipeline.
        Example: estimator_name='random_forest_clf'

    named_ranges : list of tuples
        List of tuples of size 2, wherein each element is (param_name, param_values).
        Here, a param_name is the name of an estimator's parameter, and
            param_values are an iterable of values for that parameter.

        named_ranges = [('n_estimators', [50, 100, 500]),
                        ('max_features', [10, 20, 50, 100]),
                        ('min_samples_leaf', [3, 5, 10, 20])]

    Returns
    -------
    param_grid : dict
        An sklearn.pipeline compatible dict of named parameter ranges.

    """

    if named_ranges is None or estimator_name in [None, '']:
        return None

    prepend_param_name = lambda string: '{}__{}'.format(estimator_name, string)
    param_grid = dict()
    for param_name, param_values in named_ranges:
        param_grid[prepend_param_name(param_name)] = param_values

    return param_grid


def compute_reduced_dimensionality(select_method, train_class_sizes, train_data_dim):
    """
    Estimates the number of features to retain for feature selection based on method chosen.

    Parameters
    ----------
    select_method : str
        Type of feature selection.
    train_class_sizes : iterable
        Sizes of all classes in training set.
    train_data_dim : int
        Data dimensionality for the feature set.

    Returns
    -------
    reduced_dim : int
        Reduced dimensionality n, such that 1 <= n <= train_data_dim.

    Raises
    ------
    ValueError
        If method choice is invalid.
    """

    # default to use them all.
    if select_method in [None, 'all']:
        return train_data_dim


    def do_sqrt(size):
        return np.ceil(np.sqrt(size))


    def do_tenth(size):
        return np.ceil(size / 10)


    def do_log2(size):
        return np.ceil(np.log2(size))


    get_reduced_dim = {'tenth': do_tenth,
                       'sqrt' : do_sqrt,
                       'log2' : do_log2}

    if isinstance(select_method, str):
        if select_method in get_reduced_dim:
            smallest_class_size = np.sum(train_class_sizes)
            calc_size = get_reduced_dim[select_method](smallest_class_size)
        else:
            # arg could be string coming from command line
            calc_size = np.int64(select_method)
        reduced_dim = min(calc_size, train_data_dim)
    elif isinstance(select_method, int):
        if select_method > train_data_dim:  # case of Inf is covered
            reduced_dim = train_data_dim
            print('Reducing the feature selection size to {}, '
                  'to accommondate the current feature set.'.format(train_data_dim))
        else:
            reduced_dim = select_method
    elif isinstance(select_method, float):
        if not np.isfinite(select_method):
            raise ValueError('Fraction for the reduced dimensionality must be finite within (0.0, 1.0).')
        elif select_method <= 0.0:
            reduced_dim = 1
        elif select_method >= 1.0:
            reduced_dim = train_data_dim
        else:
            reduced_dim = np.int64(np.floor(train_data_dim / select_method))
    else:
        raise ValueError(
            'Invalid method to choose size of feature selection. '
            'It can only be '
            '1) string or '
            '2) finite integer (< data dimensionality) or '
            '3) a fraction between 0.0 and 1.0 !')

    # ensuring it is an integer >= 1
    reduced_dim = np.int64(np.max([reduced_dim, 1]))
    # # the following statement would print it in each trial of CV!! Not useful
    # print('reduced dimensionality: {}'.format(reduced_dim))

    return reduced_dim


def add_new_params(old_grid, new_grid, old_name, new_name):
    """
    Adds new items (parameters) in-place to old dict (of parameters),
    ensuring no overlap in new parameters with old parameters which prevents silent overwrite.

    """

    if new_grid:
        new_params = set(new_grid.keys())
        old_params = set(old_grid.keys())
        if len(old_params.intersection(new_params)) > 0:
            raise ValueError(
                    'Overlap in parameters between {} and {} of the chosen pipeline.'.format(old_name, new_name))

        old_grid.update(new_grid)

    return


def get_DecisionTreeClassifier(reduced_dim=None, grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

        The 'light' option provides a lighter and much less exhaustive grid search to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and "folk wisdom".
        Useful to get a "very rough" idea of performance for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        splitter_strategies = ['best', 'random']
        split_criteria = ['gini', 'entropy']
        range_min_leafsize = [1, 3, 5, 10, 20]

        # if user supplied reduced_dim, it will be tried also. Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        splitter_strategies = ['best', 'random']
        split_criteria = ['gini', ]
        range_min_leafsize = [1, 5]
        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        splitter_strategies = ['best',]
        split_criteria = ['gini', ]
        range_min_leafsize = [1, ]
        range_max_features = [reduced_dim]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'decision_tree_clf'
    param_list_values = [('criterion', split_criteria),
                         ('splitter', splitter_strategies),
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    dtc = DecisionTreeClassifier(max_features=reduced_dim)

    return dtc, clf_name, param_grid


def get_ExtraTreesClassifier(reduced_dim=None, grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

        The 'light' option provides a lighter and much less exhaustive grid search to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and "folk wisdom".
        Useful to get a "very rough" idea of performance for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ['gini', 'entropy']
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also. Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_num_trees = [50, 250, ]
        split_criteria = ['gini', ]
        range_min_leafsize = [1, 5]
        range_min_impurity = [0.0, 0.01]

        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_num_trees = [250, ]
        split_criteria = ['gini', ]
        range_min_leafsize = [1, ]
        range_min_impurity = [0.0, ]

        range_max_features = [reduced_dim]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'extra_trees_clf'
    param_list_values = [('n_estimators', range_num_trees),
                         ('criterion', split_criteria),
                         # ('min_impurity_decrease',  range_min_impurity), # ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = ExtraTreesClassifier(max_features=reduced_dim,
                               n_estimators=max(range_num_trees),
                               oob_score=True,
                               bootstrap=True)

    return rfc, clf_name, param_grid


def get_svc(reduced_dim=None, grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the SVM classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

        The 'light' option provides a lighter and much less exhaustive grid search to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and "folk wisdom".
        Useful to get a "very rough" idea of performance for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        #TODO try pruning values based on their processing time/redundancy
        range_penalty = np.power(10.0, range(-3, 6))
        range_kernel = ['linear', 'poly', 'rbf']
        range_degree = [1, 2, 3]
        range_gamma = ['auto', ]
        range_gamma.extend(np.power(2.0, range(-5, 5)))
        range_coef0 = np.sort(np.hstack((np.arange(-100, 101, 50),
                                         np.arange(-1.0, 1.01, 0.25))))

        # if user supplied reduced_dim, it will be tried also. Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_penalty = np.power(10.0, range(-3, 6, 3))
        range_kernel = ['rbf']
        range_gamma = ['auto', ]
        range_gamma.extend(np.power(2.0, range(-5, 5, 3)))
        range_coef0 = np.sort(np.hstack((np.arange(-50, 101, 100),
                                         np.arange(-0.5, 1.01, 1.0))))

        range_max_features = ['sqrt', 0.25, reduced_dim]

        # setting for sake of completeness, although this will be ignored
        range_degree = [1, ]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_penalty = [10.0, ]
        range_kernel = ['rbf']
        range_gamma = ['auto', ]
        range_coef0 = [0.0, ]

        range_max_features = [reduced_dim]

        # setting for sake of completeness, although this will be ignored
        range_degree = [0, ]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    clf_name = 'svc'
    param_list_values = [('C', range_penalty),
                         ('kernel', range_kernel),
                         ('degree', range_degree),
                         ('gamma', range_gamma),
                         ('coef0', range_coef0),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    from sklearn.svm import SVC
    clf = SVC(probability=True)

    return clf, clf_name, param_grid


def get_RandomForestClassifier(reduced_dim=None, grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

        The 'light' option provides a lighter and much less exhaustive grid search to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and "folk wisdom".
        Useful to get a "very rough" idea of performance for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ['gini', 'entropy']
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also. Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_num_trees = [50, 250, ]
        split_criteria = ['gini', ]
        range_min_leafsize = [1, 5]
        range_min_impurity = [0.0, 0.01]

        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_num_trees = [250, ]
        split_criteria = ['gini', ]
        range_min_leafsize = [1, ]
        range_min_impurity = [0.0, ]

        range_max_features = [reduced_dim]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'random_forest_clf'
    param_list_values = [('n_estimators', range_num_trees),
                         ('criterion', split_criteria),
                         # ('min_impurity_decrease',  range_min_impurity), # ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = RandomForestClassifier(max_features=reduced_dim, n_estimators=max(range_num_trees), oob_score=True)

    return rfc, clf_name, param_grid


def get_xgboost(reduced_dim=None, grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the extremene gradient boosting (XGB) classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

        The 'light' option provides a lighter and much less exhaustive grid search to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and "folk wisdom".
        Useful to get a "very rough" idea of performance for different feature sets, and for debugging.

    Returns
    -------

    """

    from xgboost import XGBClassifier

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        # TODO consult literature for better selection of ranges
        range_max_depth = [2, 6, 10]
        # range_min_child_weight = []
        range_gamma = [0, 3, 5, 10]
        range_subsample = [0.5, 0.75, 1.0]

        range_colsample_bytree = [0.6, 0.8, 1.0]
        range_learning_rate = [0.15, 0.3, 0.5]

        range_num_feature = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_max_depth = [2, 6]
        # range_min_child_weight = []
        range_gamma = [0, 3, ]
        range_subsample = [0.5, 1.0]

        range_colsample_bytree = [0.6, 0.8, 1.0]
        range_learning_rate = [0.15, 0.3, ]

        range_num_feature = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_max_depth = [2, ]
        # range_min_child_weight = []
        range_gamma = [0, ]
        range_subsample = [1.0, ]

        range_colsample_bytree = [1.0, ]
        range_learning_rate = [0.3,]

        range_num_feature = [reduced_dim]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'xgboost_clf'
    param_list_values = [('max_depth', range_max_depth),
                         ('learning_rate', range_learning_rate),
                         ('gamma', range_gamma),
                         ('colsample_bytree', range_colsample_bytree),
                         ('subsample', range_subsample),
                         ('num_feature', range_num_feature),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    xgb = XGBClassifier(num_feature=reduced_dim,
                        max_depth=3,
                        subsample=0.8,
                        predictor='cpu_predictor',
                        nthread=1, # to avoid interactions with other parallel tasks
                        )

    return xgb, clf_name, param_grid



def get_classifier(classifier_name=cfg.default_classifier,
                   reduced_dim='all',
                   grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the named classifier and its parameter grid.

    Parameters
    ----------
    classifier_name : str
        String referring to a valid scikit-learn classifier.

    reduced_dim : int or str
        Reduced dimensionality, either an integer or "all", which defaults to using everything.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

    Returns
    -------
    clf : sklearn.estimator
        Valid scikit-learn estimator.
    clf_name : str
        String identifying the classifier to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    classifier_name = classifier_name.lower()
    map_to_method = dict(randomforestclassifier=get_RandomForestClassifier,
                         extratreesclassifier=get_ExtraTreesClassifier,
                         decisiontreeclassifier=get_DecisionTreeClassifier,
                         svm=get_svc,
                         xgboost=get_xgboost)

    if classifier_name not in map_to_method:
        raise NotImplementedError('Invalid name or classifier not implemented.')

    clf_builder = map_to_method[classifier_name]
    clf, clf_name, param_grid = clf_builder(reduced_dim, grid_search_level)

    return clf, clf_name, param_grid


def get_feature_selector(feat_selector_name='variancethreshold',
                         reduced_dim='all'):
    """
    Returns the named classifier and its parameter grid.

    Parameters
    ----------
    feat_selector_name : str
        String referring to a valid scikit-learn feature selector.
    reduced_dim : str or int
        Reduced dimensionality, either an integer or "all", which defaults to using everything.

    Returns
    -------
    feat_selector : sklearn.feature_selection method
        Valid scikit-learn feature selector.
    clf_name : str
        String identifying the feature selector to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    fs_name = feat_selector_name.lower()
    if fs_name in ['selectkbest_mutual_info_classif', ]:
        # no param optimization for feat selector for now.
        feat_selector = SelectKBest(score_func=mutual_info_classif, k=reduced_dim)
        # TODO optimize the num features to select as part of grid search
        fs_param_grid = None

    elif fs_name in ['selectkbest_f_classif', ]:
        # no param optimization for feat selector for now.
        feat_selector = SelectKBest(score_func=f_classif, k=reduced_dim)
        fs_param_grid = None

    elif fs_name in ['variancethreshold', ]:
        feat_selector = VarianceThreshold(threshold=cfg.variance_threshold)
        fs_param_grid = None

    else:
        raise NotImplementedError('Invalid name or feature selector not implemented.')

    return feat_selector, fs_name, fs_param_grid


def get_preprocessor(preproc_name='RobustScaler'):
    """
    Returns a requested preprocessor
    """

    from sklearn.preprocessing import RobustScaler

    approved_preprocessor_list = map(str.lower, dir(sklearn.preprocessing))

    preproc_name = preproc_name.lower()
    if preproc_name not in approved_preprocessor_list:
        raise ValueError('chosen preprocessor not supported.')

    if preproc_name in ['robustscaler']:
        preproc = RobustScaler(with_centering=True, with_scaling=True, quantile_range=cfg.robust_scaler_iqr)
        param_grid = None
    else:
        # TODO returning preprocessor blindly without any parameters
        preproc = getattr(sklearn.preprocessing, preproc_name)
        param_grid = None

    return preproc, preproc_name, param_grid


def get_pipeline(train_class_sizes, feat_sel_size, num_features,
                 preprocessor_name='robustscaler',
                 feat_selector_name=cfg.default_feat_select_method,
                 classifier_name=cfg.default_classifier,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Constructor for pipeline (feature selection followed by a classifier).

    Parameters
    ----------
    train_class_sizes : list
        sizes of different classes in the training set, to estimate the reduced dimensionality.

    feat_sel_size : str or int
        Method specify the method to estimate the reduced dimensionality.
        Choices : integer, or one of the methods in ['sqrt', 'tenth', 'log2', 'all']

    num_features : int
        Number of features in the training set.

    classifier_name : str
        String referring to a valid scikit-learn classifier.

    feat_selector_name : str
        String referring to a valid scikit-learn feature selector.

    preprocessor_name : str
        String referring to a valid scikit-learn preprocessor
        (This can technically be another feature selector, although discourage).

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Valid scikit learn pipeline.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.
        Grid contains ranges for parameters of all the steps, including preprocessor, feature selector and classifier.

    """

    reduced_dim = compute_reduced_dimensionality(feat_sel_size, train_class_sizes, num_features)

    preproc, preproc_name, preproc_param_grid = get_preprocessor(preprocessor_name)
    estimator, est_name, clf_param_grid = get_classifier(classifier_name, reduced_dim, grid_search_level)
    feat_selector, fs_name, fs_param_grid = get_feature_selector(feat_selector_name, reduced_dim)

    # composite grid of parameters from all steps
    param_grid = clf_param_grid.copy()
    add_new_params(param_grid, preproc_param_grid, est_name, preproc_name)
    add_new_params(param_grid, fs_param_grid, est_name, fs_name)

    steps = [(preproc_name, preproc),
             (fs_name, feat_selector),
             (est_name, estimator)]
    pipeline = Pipeline(steps)

    return pipeline, param_grid


def get_feature_importance(clf_name, clf, num_features, index_selected_features, fill_value=np.nan):
    "Extracts the feature importance of input features, if available."

    attr_importance = {'randomforestclassifier': 'feature_importances_',
                       'extratreesclassifier'  : 'feature_importances_',
                       'decisiontreeclassifier': 'feature_importances_',
                       'svm'                   : 'coef_',
                       'xgboost'               : 'feature_importances_',}

    feat_importance = np.full(num_features, fill_value)
    if hasattr(clf, attr_importance[clf_name]):
        feat_importance[index_selected_features] = getattr(clf, attr_importance[clf_name])

    return feat_importance
