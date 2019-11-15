"""

Module defining methods and classes needed to manage results produced.

"""

import numpy as np
from neuropredict import config_neuropredict as cfg
from abc import abstractmethod
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring

class CVResults(object):
    """
    Class to store and organize the results for a CV run.
    """

    def __init__(self,
                 estimator=cfg.default_classifier,
                 metric_set=(cfg.default_scoring_metric, )):
        "Constructor."

        self.metric_set = _check_multimetric_scoring(estimator, metric_set)
        self.metric_val = {name: dict() for name in self.metric_set.keys()}

        self._attr = dict()


    def add(self, run_id, dataset_id, predicted, true_targets):
        """
        Method to populate the CVResults class with predictions/accuracies,
         coming from different repetitions of CV.
        """

        for name, score_func in self.metric_set.items():
            self.metric_val[name][(run_id, dataset_id)] = score_func(
                    true_targets, predicted)

    def add_attr(self, run_id, dataset_id, name, value):
        """
        Method to store miscellaneous attributes for post-hoc analyses,
        """

        if (run_id, dataset_id) not in self._attr:
            self._attr[(run_id, dataset_id)] = dict()

        self._attr[(run_id, dataset_id)][name] = value


    def save(self):
        "Method to persist the results to disk."

    def load(self):
        "Method to load previously saved results e.g. to redo visualizations"

    @abstractmethod
    def export(self):
        "Method to export the results to different formats (e.g. pyradigm or CSV)"


class ClassifyCVResults(CVResults):
    """Custom CVResults class to accommodate classification-specific evaluation."""

    def __init__(self,
                 estimator=cfg.default_classifier,
                 metric_set=cfg.default_metric_set_classification):
        "Constructor."

        super().__init__(estimator=estimator, metric_set=metric_set)

        self._conf_mat = dict() # confusion matrix
        self._misclf_samplets = dict() # list of misclassified samplets


    def add_diagnostics(self, run_id, dataset_id, conf_mat, misclfd_ids):
        """Method to save the confusion matrix from each prediction run"""

        self._conf_mat[(run_id, dataset_id)] = conf_mat
        self._misclf_samplets[(run_id, dataset_id)] = misclfd_ids


    def export(self):
        """Method to export the results in portable formats reusable outside this
        library"""

        raise NotImplementedError()


class RegressCVResults(CVResults):
    """Custom CVResults class to accommodate classification-specific evaluation."""


    def __init__(self,
                 estimator=cfg.default_regressor,
                 metric_set=cfg.default_metric_set_regression):
        "Constructor."

        super().__init__(estimator=estimator, metric_set=metric_set)