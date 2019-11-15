"""Module to define classification-oriented workflows and methods"""


from neuropredict import config_neuropredict as cfg
from neuropredict.base import BaseWorkflow
from neuropredict.utils import uniquify_in_order
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

class ClassificationWorkflow(BaseWorkflow):
    """
    Class defining an neuropredict experiment to be run.

    Encapsulates all the details necessary for execution,
        hence easing the save/load/decide workflow.

    """


    def __init__(self,
                 datasets,
                 pred_model=cfg.default_classifier,
                 impute_strategy=cfg.default_imputation_strategy,
                 dim_red_method=cfg.default_dim_red_method,
                 reduced_dim=cfg.default_reduced_dim_size,
                 train_perc=cfg.default_train_perc,
                 num_rep_cv=cfg.default_num_repetitions,
                 scoring=cfg.default_scoring_metric,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                 num_procs=cfg.DEFAULT_NUM_PROCS,
                 user_options=None,
                 checkpointing=cfg.default_checkpointing):

        super().__init__(datasets,
                         pred_model=pred_model,
                         impute_strategy=impute_strategy,
                         dim_red_method=dim_red_method,
                         reduced_dim=reduced_dim,
                         train_perc=train_perc,
                         num_rep_cv=num_rep_cv,
                         scoring=scoring,
                         grid_search_level=grid_search_level,
                         num_procs=num_procs,
                         user_options=user_options,
                         checkpointing=checkpointing)

        self._target_set = list(uniquify_in_order(self.datasets.targets.values()))
        # set does not preserve order, sorting makes it stable across sessions

    def _eval_predictions(self, pipeline, test_data, true_targets, run_id, ds_id):
        """Evaluate predictions and perf estimates to results class.

        Prints a quick summary too, as an indication of progress.
        """

        predicted_targets = pipeline.predict(test_data)
        self.results.add(run_id, ds_id, predicted_targets, true_targets)

        predict_proba_name = 'predict_proba'
        if hasattr(pipeline, predict_proba_name):
            predicted_prob = pipeline.predict_proba(test_data)
            self.results.add_attr(run_id, ds_id, predict_proba_name, predicted_prob)

        conf_mat = confusion_matrix(true_targets, predicted_targets,
                                    labels=self._target_set) # to control row order
        self.results.add_diagnostics(conf_mat,
                                     true_targets[predicted_targets!=true_targets])



