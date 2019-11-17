"""

Module defining methods and classes needed to manage results produced.

"""

from abc import abstractmethod
import numpy as np
from neuropredict import config_neuropredict as cfg
from neuropredict.algorithms import get_estimator_by_name
from neuropredict.utils import is_iterable_but_not_str


class CVResults(object):
    """
    Class to store and organize the results for a CV run.
    """


    def __init__(self,
                 estimator_name=cfg.default_classifier,
                 metric_set=(cfg.default_scoring_metric,)):
        "Constructor."

        estimator = get_estimator_by_name(estimator_name)
        if is_iterable_but_not_str(metric_set):
            self.metric_set = {func.__name__: func for func in metric_set}
        self.metric_val = {name: dict() for name in self.metric_set.keys()}

        self._count = 0
        self._attr = dict()


    def add(self, run_id, dataset_id, predicted, true_targets):
        """
        Method to populate the CVResults class with predictions/accuracies,
         coming from different repetitions of CV.
        """

        msgs = list()
        msgs.append('CV run {:<3} dataset {:<20} :'.format(run_id, dataset_id))
        for name, score_func in self.metric_set.items():
            score = score_func(true_targets, predicted)
            self.metric_val[name][(run_id, dataset_id)] = score
            msgs.append(' {:>20} {:8.3f}'.format(name, score))

        # quick summary print
        print(' '.join(msgs))

        # counting
        self._count += 1


    def add_attr(self, run_id, dataset_id, name, value):
        """
        Method to store miscellaneous attributes for post-hoc analyses,
        """

        if (run_id, dataset_id) not in self._attr:
            self._attr[(run_id, dataset_id)] = dict()

        self._attr[(run_id, dataset_id)][name] = value


    def _metric_summary(self):
        """for FYI"""

        if self._count > 0:
            summary = list()
            for metric, distr in self.metric_val.items():
                median = np.median(distr)
                SD = np.std(distr)
                summary.append(' {} : median {} SD {}'.format(metric, median, SD))
            return '\n'.join(summary)
        else:
            return 'No results added so far!'


    def __str__(self):
        """Simple summary"""

        return 'Metrics : {}\n # runs : {}\n {}' \
               ''.format(', '.join(self.metric_set.keys()), self._count,
                         self._metric_summary())

    def __repr__(self):
        return self.__str__()

    def __format__(self, format_spec):
        return self.__str__()

    def save(self):
        "Method to persist the results to disk."


    def load(self):
        "Method to load previously saved results e.g. to redo visualizations"


    @abstractmethod
    def dump(self, out_dir):
        """Method for quick dump, for checkpointing purposes"""


    @abstractmethod
    def export(self):
        "Method to export the results to different formats (e.g. pyradigm or CSV)"


class ClassifyCVResults(CVResults):
    """Custom CVResults class to accommodate classification-specific evaluation."""


    def __init__(self,
                 estimator=cfg.default_classifier,
                 metric_set=cfg.default_metric_set_classification):
        "Constructor."

        super().__init__(estimator_name=estimator, metric_set=metric_set)

        self._conf_mat = dict()  # confusion matrix
        self._misclf_samplets = dict()  # list of misclassified samplets


    def add_diagnostics(self, run_id, dataset_id, conf_mat, misclfd_ids):
        """Method to save the confusion matrix from each prediction run"""

        self._conf_mat[(run_id, dataset_id)] = conf_mat
        self._misclf_samplets[(run_id, dataset_id)] = misclfd_ids


    def export(self):
        """Method to export the results in portable formats reusable outside this
        library"""

        raise NotImplementedError()


    def dump(self, out_dir):
        """Method for quick dump, for checkpointing purposes"""

        from os.path import join as pjoin, exists
        from neuropredict.utils import make_time_stamp
        import pickle
        out_path = pjoin(out_dir, 'cv_results_quick_dump_{}.pkl'
                                  ''.format(make_time_stamp()))
        if exists(out_path):
            from os import remove
            remove(out_path)
        with open(out_path, 'wb') as df:
            to_save = [self.metric_set, self.metric_val, self._attr,
                       self._conf_mat, self._misclf_samplets]
            pickle.dump(to_save, df)


class RegressCVResults(CVResults):
    """Custom CVResults class to accommodate classification-specific evaluation."""


    def __init__(self,
                 estimator=cfg.default_regressor,
                 metric_set=cfg.default_metric_set_regression):
        "Constructor."

        super().__init__(estimator_name=estimator, metric_set=metric_set)


    def dump(self, out_dir):
        """Method for quick dump, for checkpointing purposes"""

        from os.path import join as pjoin, exists
        from neuropredict.utils import make_time_stamp
        import pickle
        out_path = pjoin(out_dir, 'cv_results_quick_dump_{}.pkl'
                                  ''.format(make_time_stamp()))
        if exists(out_path):
            from os import remove
            remove(out_path)
        with open(out_path, 'wb') as df:
            to_save = [self.metric_set, self.metric_val, self._attr]
            pickle.dump(to_save, df)
