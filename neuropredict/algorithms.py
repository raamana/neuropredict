import numpy as np
import sklearn
from scipy.sparse import issparse
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import (SelectKBest, VarianceThreshold, f_classif,
                                       mutual_info_classif)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from neuropredict import config as cfg


def get_estimator_by_name(est_name):
    """Returns an usable sklearn Estimator identified by name"""

    map_to_method = dict(randomforestclassifier=RandomForestClassifier,
                         extratreesclassifier=ExtraTreesClassifier,
                         decisiontreeclassifier=DecisionTreeClassifier,
                         svm=SVC,
                         svr=SVR,
                         randomforestregressor=RandomForestRegressor,
                         extratreesregressor=ExtraTreesRegressor,
                         decisiontreeregressor=DecisionTreeRegressor,
                         gaussianprocessregressor=GaussianProcessRegressor,
                         gradientboostingregressor=GradientBoostingRegressor,
                         kernelridge=KernelRidge,
                         bayesianridge=BayesianRidge,
                         )

    return map_to_method.get(est_name.lower(), 'Estimator_Not_Processed_Yet')


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


def compute_reduced_dimensionality(size_spec, train_set_size, train_data_dim):
    """
    Estimates the number of features to retain for feature selection.

    Parameters
    ----------
    size_spec : str
        Type of feature selection.
    train_set_size : iterable
        Total size of the training set.
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
    if size_spec in [None, 'all']:
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

    if isinstance(size_spec, str):
        if size_spec in get_reduced_dim:
            calc_size = get_reduced_dim[size_spec](train_set_size)
        else:
            # arg could be string coming from command line
            calc_size = np.int64(size_spec)
        reduced_dim = min(calc_size, train_data_dim)
    elif isinstance(size_spec, int):
        if size_spec > train_data_dim:  # case of Inf is covered
            reduced_dim = train_data_dim
            print('Reducing the feature selection size to {}, '
                  'to accommodate the current feature set.'.format(train_data_dim))
        else:
            reduced_dim = size_spec
    elif isinstance(size_spec, float):
        if not np.isfinite(size_spec):
            raise ValueError('Fraction for the reduced dimensionality '
                             'must be finite within (0.0, 1.0).')
        elif size_spec <= 0.0:
            reduced_dim = 1
        elif size_spec >= 1.0:
            reduced_dim = train_data_dim
        else:
            reduced_dim = np.int64(np.floor(train_data_dim * size_spec))
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
        ensuring no overlap in new parameters with old parameters
        which prevents silent overwrite.

    """

    if new_grid:
        new_params = set(new_grid.keys())
        old_params = set(old_grid.keys())
        if len(old_params.intersection(new_params)) > 0:
            raise ValueError('Overlap in parameters between {} and {} of the ' \
                             'chosen pipeline.'.format(old_name, new_name))

        old_grid.update(new_grid)

    return


def get_DecisionTreeClassifier(reduced_dim=None,
                               grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        splitter_strategies = ['best', 'random']
        split_criteria = ['gini', 'entropy']
        range_min_leafsize = [1, 3, 5, 10, 20]

        # if user supplied reduced_dim, it will be tried also.
        #   Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        splitter_strategies = ['best', 'random']
        split_criteria = ['gini', ]
        range_min_leafsize = [1, 5]
        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        splitter_strategies = ['best', ]
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


def get_ExtraTreesClassifier(reduced_dim=None,
                             grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ['gini', 'entropy', 'log_loss']
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also.
        # Default None --> all features.
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
                         # ('min_impurity_decrease',  range_min_impurity),
                         #  ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = ExtraTreesClassifier(max_features=reduced_dim,
                               n_estimators=max(range_num_trees),
                               oob_score=True,
                               bootstrap=True)

    return rfc, clf_name, param_grid


def get_ExtraTreesRegressor(reduced_dim=None,
                            grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Extra Trees Regressor and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ['squared_error', 'absolute_error']
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also.
        # Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_num_trees = [50, 250, ]
        split_criteria = ['squared_error', ]
        range_min_leafsize = [1, 5]
        range_min_impurity = [0.0, 0.01]

        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_num_trees = [250, ]
        split_criteria = ['squared_error', ]
        range_min_leafsize = [1, ]
        range_min_impurity = [0.0, ]

        range_max_features = [reduced_dim]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'extra_trees_regr'
    param_list_values = [('n_estimators', range_num_trees),
                         ('criterion', split_criteria),
                         # ('min_impurity_decrease',  range_min_impurity),
                         #  ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = ExtraTreesRegressor(max_features=reduced_dim,
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
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        # TODO try pruning values based on their processing time/redundancy
        range_penalty = np.power(10.0, range(-3, 6))
        range_kernel = ['linear', 'poly', 'rbf']
        range_degree = [1, 2, 3]
        range_gamma = ['auto', ]
        range_gamma.extend(np.power(2.0, range(-5, 5)))
        range_coef0 = np.sort(np.hstack((np.arange(-100, 101, 50),
                                         np.arange(-1.0, 1.01, 0.25))))

    elif grid_search_level in ['light']:
        range_penalty = np.power(10.0, range(-3, 5, 1))
        range_kernel = ['rbf']
        range_gamma = list()  # ['auto', ]
        range_gamma.extend(np.power(2.0, range(-5, 4, 1)))
        range_coef0 = [0.0, ]

        # setting for sake of completeness, although this will be ignored
        range_degree = [1, ]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_penalty = [10.0, ]
        range_kernel = ['rbf']
        range_gamma = ['auto', ]
        range_coef0 = [0.0, ]

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

    clf = SVC(probability=True)

    return clf, clf_name, param_grid


def get_RandomForestClassifier(reduced_dim=None,
                               grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ['gini', 'entropy', 'log_loss']
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also.
        # Default None --> all features.
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
                         # ('min_impurity_decrease',  range_min_impurity),
                         # ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = RandomForestClassifier(max_features=reduced_dim,
                                 n_estimators=max(range_num_trees),
                                 oob_score=True)

    return rfc, clf_name, param_grid


def _get_xgboost_params_ranges(grid_search_level):
    """"""

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        # TODO consult literature for better selection of ranges
        range_max_depth = [2, 6, 10]
        # range_min_child_weight = []
        range_gamma = [0, 3, 5, 10]
        range_subsample = [0.5, 0.75, 1.0]

        range_colsample_bytree = [0.6, 0.8, 1.0]
        range_learning_rate = [0.15, 0.3, 0.5]

    elif grid_search_level in ['light']:
        range_max_depth = [2, 6]
        # range_min_child_weight = []
        range_gamma = [0, 3, ]
        range_subsample = [0.5, 1.0]

        range_colsample_bytree = [0.6, 0.8, 1.0]
        range_learning_rate = [0.15, 0.3, ]

    elif grid_search_level in ['none']:  # single point on the hyperparameter grid
        range_max_depth = [2, ]
        # range_min_child_weight = []
        range_gamma = [0, ]
        range_subsample = [1.0, ]

        range_colsample_bytree = [1.0, ]
        range_learning_rate = [0.3, ]

    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    param_list_values = [('max_depth', range_max_depth),
                         ('learning_rate', range_learning_rate),
                         ('gamma', range_gamma),
                         ('colsample_bytree', range_colsample_bytree),
                         ('subsample', range_subsample),
                         ]

    return param_list_values


def get_xgboost(reduced_dim=None,
                grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the extremene gradient boosting (XGB) classifier and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.
    Returns
    -------

    """

    from xgboost import XGBClassifier
    est_name = 'xgboost_clf'
    param_grid = make_parameter_grid(est_name,
                                     _get_xgboost_params_ranges(grid_search_level))
    xgb = XGBClassifier(num_feature=reduced_dim,
                        max_depth=3,
                        subsample=0.8,
                        predictor='cpu_predictor',
                        n_jobs=1,  # to avoid interactions with other parallel tasks
                        )

    return xgb, est_name, param_grid


def get_xgboostregressor(reduced_dim=None,
                         grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the extremene gradient boosting (XGB) regressor and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.
    Returns
    -------

    """

    est_name = 'xgb_regr'
    param_grid = make_parameter_grid(est_name,
                                     _get_xgboost_params_ranges(grid_search_level))
    from xgboost import XGBRegressor
    xgb = XGBRegressor(objective='reg:squarederror',
                       num_feature=reduced_dim,
                       max_depth=3,
                       subsample=0.8,
                       predictor='cpu_predictor',
                       n_jobs=1,  # to avoid interactions with other parallel tasks
                       )

    return xgb, est_name, param_grid


def get_estimator(est_name=cfg.default_classifier,
                  reduced_dim='all',
                  grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the named classifier and its parameter grid.

    Parameters
    ----------
    est_name : str
        String referring to a valid scikit-learn classifier.

    reduced_dim : int or str
        Reduced dimensionality, either an integer or
        "all", which defaults to using everything.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------
    clf : sklearn.estimator
        Valid scikit-learn estimator.
    clf_name : str
        String identifying the classifier to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    est_name = est_name.lower()
    map_to_method = dict(randomforestclassifier=get_RandomForestClassifier,
                         extratreesclassifier=get_ExtraTreesClassifier,
                         decisiontreeclassifier=get_DecisionTreeClassifier,
                         svm=get_svc,
                         xgboost=get_xgboost,
                         randomforestregressor=get_RandomForestRegressor,
                         extratreesregressor=get_ExtraTreesRegressor,
                         gradientboostingregressor=get_GradientBoostingRegressor,
                         xgboostregressor=get_xgboostregressor)

    if est_name not in map_to_method:
        raise NotImplementedError('Invalid name for estimator: {}\n'
                                  'Implemented estimators: {}'
                                  ''.format(est_name, list(map_to_method.keys())))

    est_builder = map_to_method[est_name]
    est, est_name, param_grid = est_builder(reduced_dim, grid_search_level)

    return est, est_name, param_grid


def get_dim_reducer(total_num_samplets,
                    dr_name='variancethreshold',
                    reduced_dim='all'):
    """
    Returns the named dimensionality reduction method and its parameter grid.

    Parameters
    ----------
    dr_name : str
        String referring to a valid scikit-learn feature selector.
    reduced_dim : str or int
        Reduced dimensionality, either an integer
        or "all", which defaults to using everything.

    Returns
    -------
    dim_red : sklearn.feature_selection method
        Valid scikit-learn feature selector.
    clf_name : str
        String identifying the feature selector to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    # TODO not optimizing hyper params for any technique: Isomap, LLE etc

    dr_name = dr_name.lower()
    if dr_name in ['isomap', ]:
        from sklearn.manifold import Isomap
        dim_red = Isomap(n_components=reduced_dim)
        dr_param_grid = None
    elif dr_name in ['lle', ]:
        from sklearn.manifold import LocallyLinearEmbedding
        dim_red = LocallyLinearEmbedding(n_components=reduced_dim,
                                         method='standard')
        dr_param_grid = None
    elif dr_name in ['lle_modified', ]:
        from sklearn.manifold import LocallyLinearEmbedding
        dim_red = LocallyLinearEmbedding(n_components=reduced_dim,
                                         method='modified')
        dr_param_grid = None
    elif dr_name in ['lle_hessian', ]:
        from sklearn.manifold import LocallyLinearEmbedding

        n_components = reduced_dim
        # ensuring n_neighbors meets the required magnitude
        dp = n_components * (n_components + 1) // 2
        n_neighbors = n_components + dp + 1
        n_neighbors = min(n_neighbors, total_num_samplets)
        dim_red = LocallyLinearEmbedding(n_components=n_components,
                                         n_neighbors=n_neighbors,
                                         method='hessian')
        dr_param_grid = None
    elif dr_name in ['lle_ltsa', ]:
        from sklearn.manifold import LocallyLinearEmbedding
        dim_red = LocallyLinearEmbedding(n_components=reduced_dim,
                                         method='ltsa')
        dr_param_grid = None
    elif dr_name in ['selectkbest_mutual_info_classif', ]:
        # no param optimization for feat selector for now.
        dim_red = SelectKBest(score_func=mutual_info_classif, k=reduced_dim)
        dr_param_grid = None

    elif dr_name in ['selectkbest_f_classif', ]:
        # no param optimization for feat selector for now.
        dim_red = SelectKBest(score_func=f_classif, k=reduced_dim)
        dr_param_grid = None

    elif dr_name in ['variancethreshold', ]:
        dim_red = VarianceThreshold(threshold=cfg.variance_threshold)
        dr_param_grid = None

    else:
        raise ValueError('Invalid name, or method {} not implemented.\n'
                         'Choose one of {}'.format(dr_name,
                                                   cfg.all_dim_red_methods))

    return dim_red, dr_name, dr_param_grid


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
        preproc = RobustScaler(with_centering=True, with_scaling=True,
                               quantile_range=cfg.robust_scaler_iqr)
        param_grid = None
    else:
        # TODO returning preprocessor blindly without any parameters
        preproc = getattr(sklearn.preprocessing, preproc_name)
        param_grid = None

    return preproc, preproc_name, param_grid


def get_pipeline(train_class_sizes, feat_sel_size, num_features,
                 preproc_name='robustscaler',
                 fsr_name=cfg.default_dim_red_method,
                 clfr_name=cfg.default_classifier,
                 gs_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Constructor for pipeline (feature selection followed by a classifier).

    Parameters
    ----------
    train_class_sizes : list
        sizes of different classes in the training set,
        to estimate the reduced dimensionality.

    feat_sel_size : str or int
        Method specify the method to estimate the reduced dimensionality.
        Choices : integer, or one of the methods in ['sqrt', 'tenth', 'log2', 'all']

    num_features : int
        Number of features in the training set.

    clfr_name : str
        String referring to a valid scikit-learn classifier.

    fsr_name : str
        String referring to a valid scikit-learn feature selector.

    preproc_name : str
        String referring to a valid scikit-learn preprocessor
        (This can technically be another feature selector, although discourage).

    gs_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for
        optimization.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Valid scikit learn pipeline.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.
        Grid contains ranges for parameters of all the steps,
        including preprocessor, feature selector and classifier.

    """

    reduced_dim = compute_reduced_dimensionality(feat_sel_size,
                                                 np.sum(train_class_sizes),
                                                 num_features)

    preproc, preproc_name, preproc_param_grid = get_preprocessor(preproc_name)
    estimator, est_name, clf_param_grid = get_estimator(clfr_name, reduced_dim,
                                                        gs_level)
    feat_selector, fs_name, fs_param_grid = get_dim_reducer(
            train_class_sizes, fsr_name, reduced_dim)

    # composite grid of parameters from all steps
    param_grid = clf_param_grid.copy()
    add_new_params(param_grid, preproc_param_grid, est_name, preproc_name)
    add_new_params(param_grid, fs_param_grid, est_name, fs_name)

    steps = [(preproc_name, preproc),
             (fs_name, feat_selector),
             (est_name, estimator)]
    pipeline = Pipeline(steps)

    return pipeline, param_grid


def get_feature_importance(est_name, est, dim_red,
                           num_features, fill_value=np.nan):
    "Extracts the feature importance of input features, if available."

    feat_importance = np.full(num_features, fill_value)

    if hasattr(dim_red, 'get_support'):  # nonlinear dim red won't have this
        index_selected_features = dim_red.get_support(indices=True)

        if hasattr(est, cfg.importance_attr[est_name]):
            feat_importance[index_selected_features] = \
                getattr(est, cfg.importance_attr[est_name])

    return feat_importance


def make_pipeline(pred_model,
                  dim_red_method,
                  reduced_dim,
                  train_set_size,
                  preproc_name=cfg.default_preprocessing_method,
                  gs_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """Constructor for sklearn pipeline. Generic version of get_pipeline"""

    # preproc, preproc_name, preproc_param_grid = get_preprocessor(preproc_name)
    estimator, est_name, est_param_grid = get_estimator(pred_model, reduced_dim,
                                                        gs_level)
    dim_reducer, dr_name, dr_param_grid = get_dim_reducer(train_set_size,
                                                          dim_red_method,
                                                          reduced_dim)
    # composite grid of parameters from all steps
    param_grid = est_param_grid.copy()
    # add_new_params(param_grid, preproc_param_grid, est_name, preproc_name)
    # add_new_params(param_grid, deconf_param_grid, est_name, deconf_name)
    add_new_params(param_grid, dr_param_grid, est_name, dr_name)

    steps = [
        # (preproc_name, preproc),
        # (deconf_name, deconf),
        (dr_name, dim_reducer),
        (est_name, estimator)]
    pipeline = Pipeline(steps)
    return pipeline, param_grid


def get_deconfounder(xfm_name, grid_search_level=None):
    """Returns a valid sklearn transformer for deconfounding."""

    xfm_name = xfm_name.lower()
    if xfm_name in ('residualize', 'regressout',
                    'residualize_linear', 'regressout_linear'):
        from confounds.base import Residualize
        xfm = Residualize()
        param_list_values = []
    # elif name in ('residualize_ridge', 'residualize_kernelridge'):
    #     from confounds.base import Residualize
    #     xfm =  Residualize(model='KernelRidge')
    #     param_list_values = [('param_1', range_param1),
    #                          ('criterion_2', criteria),
    #                          ]
    # elif name in ('residualize_gpr', 'residualize_gaussianprocessregression'):
    #     from confounds.base import Residualize
    #     xfm =  Residualize(model='GPR')
    #     param_list_values = [('param_1', range_param1),
    #                          ('criterion_2', criteria),
    #                          ]
    elif xfm_name in ('augment', 'pad'):
        from confounds.base import Augment
        xfm = Augment()
        param_list_values = []
    elif xfm_name in ('dummy', 'passthrough'):
        from confounds.base import DummyDeconfounding
        xfm = DummyDeconfounding()
        param_list_values = []
    else:
        raise ValueError('Unrecognized model name! '
                         'Choose one of Residualize, Augment or Dummy.')

    param_grid = make_parameter_grid(xfm_name, param_list_values)
    return xfm, xfm_name, param_grid


def get_RandomForestRegressor(reduced_dim=None,
                              grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Random Forest regressor and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [50, 120, 350, 500]
        split_criteria = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        range_min_leafsize = [1, 3, 5, 10, 20]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also.
        # Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]

    elif grid_search_level in ['light']:
        range_num_trees = [250, 500]
        split_criteria = ['squared_error', ]
        range_min_leafsize = [1, 2]
        range_min_impurity = [0.0, 0.01]

        range_max_features = ['sqrt', 0.25, reduced_dim]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_num_trees = [250, ]
        split_criteria = ['squared_error', ]
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
                         # ('min_impurity_decrease',  range_min_impurity),
                         # ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ]
    param_grid = make_parameter_grid(clf_name, param_list_values)

    rfc = RandomForestRegressor(max_features=reduced_dim,
                                n_estimators=max(range_num_trees),
                                oob_score=True)

    return rfc, clf_name, param_grid


def get_GradientBoostingRegressor(reduced_dim=None,
                                  grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT):
    """
    Returns the Gradient Boosted Regressor and its parameter grid.

    Parameters
    ----------
    reduced_dim : int
        One of the dimensionalities to be tried.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be used for
        optimization.

        The 'light' option provides a lighter and much less exhaustive grid search
        to speed up optimization.
        Parameter values will be chosen based on defaults, previous studies and
        "folk wisdom". Useful to get a "very rough" idea of performance
        for different feature sets, and for debugging.

    Returns
    -------

    """

    grid_search_level = grid_search_level.lower()
    if grid_search_level in ['exhaustive']:
        range_num_trees = [100, 500]
        losses = ['squared_error', 'absolute_error', 'huber']  # TODO 'quantile`
        split_criteria = ['friedman_mse', 'squared_error', ]
        range_min_leafsize = [1, 2, 5, 10]
        range_min_impurity = [0.01, 0.1, 0.2]  # np.arange(0., 0.41, 0.1)

        # if user supplied reduced_dim, it will be tried also.
        # Default None --> all features.
        range_max_features = ['sqrt', 'log2', 0.25, 0.4, reduced_dim]
        n_iter_no_change = [5, ]
        validation_fraction = [0.1, ]

    elif grid_search_level in ['light']:
        range_num_trees = [250, ]
        losses = ['squared_error', 'absolute_error', 'huber']
        split_criteria = ['friedman_mse', ]
        range_min_leafsize = [1, 5]
        range_min_impurity = [0.0, 0.01]

        range_max_features = ['sqrt', reduced_dim]
        n_iter_no_change = [5, ]
        validation_fraction = [0.1, ]

    elif grid_search_level in ['none']:  # single point on the hyper-parameter grid
        range_num_trees = [250, ]
        losses = ['huber', ]
        split_criteria = ['friedman_mse', ]
        range_min_leafsize = [1, ]
        range_min_impurity = [0.0, ]

        range_max_features = [reduced_dim]
        n_iter_no_change = [5, ]
        validation_fraction = [0.1, ]
    else:
        raise ValueError('Unrecognized option to set level of grid search.')

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    est_name = 'GBR'
    param_list_values = [('n_estimators', range_num_trees),
                         ('loss', losses),
                         ('criterion', split_criteria),
                         # ('min_impurity_decrease',  range_min_impurity),
                         # ignoring this
                         ('min_samples_leaf', range_min_leafsize),
                         ('max_features', range_max_features),
                         ('n_iter_no_change', n_iter_no_change),
                         ('validation_fraction', validation_fraction),
                         ]
    param_grid = make_parameter_grid(est_name, param_list_values)

    gbr = GradientBoostingRegressor(max_features=reduced_dim,
                                    n_estimators=max(range_num_trees))

    return gbr, est_name, param_grid


def encode(train_list, test_list, dtypes):
    """
    Utility to help encode/convert data types, learning only from training set.
    Often from categorical to numerical data.
    """

    encoders = list()
    for ix, (train, test, dtype) in enumerate(zip(train_list, test_list, dtypes)):
        train = train.reshape(-1, 1)
        test = test.reshape(-1, 1)
        if not np.issubdtype(dtype, np.number):
            # passing in the full spectrum to avoid unknown category error
            var_spectrum = [np.unique(np.vstack((train, test))), ]
            enc = OneHotEncoder(categories=var_spectrum)
            enc.fit(train)
            train = enc.transform(train)
            # propagating encoding from training to test
            test = enc.transform(test)
            encoders.append(enc)
        else:
            encoders.append(None)

        if issparse(train):
            train = train.todense()

        if issparse(test):
            test = test.todense()

        train_list[ix] = train
        test_list[ix] = test

    return train_list, test_list, encoders
