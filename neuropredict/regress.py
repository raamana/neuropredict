from __future__ import print_function

import os
import textwrap
from os.path import join as pjoin

import numpy as np
# the order of import is very important to avoid circular imports
from neuropredict import __version__, config_neuropredict as cfg
from neuropredict.base import BaseWorkflow, get_parser_base, parse_common_args
from neuropredict.datasets import detect_missing_data, load_datasets
from neuropredict.utils import (check_covariates,
                                check_regressor, median_of_medians)
from neuropredict.visualize import compare_distributions, multi_scatter_plot


def get_parser_regress():
    """"""

    parser, user_feat_args, cv_args, pipeline_args, vis_args, comp_args = \
        get_parser_base()

    help_regressor = textwrap.dedent("""

    String specifying one of the implemented regressors. 
    (Regressors are carefully chosen to allow for the comprehensive report 
    provided by neuropredict).

    Default: 'RandomForestRegressor'

    """)

    pipeline_args.add_argument("-e", "--regressor", action="store",
                               dest="regressor",
                               default=cfg.default_regressor, help=help_regressor,
                               choices=cfg.regressor_choices, type=str.lower)

    return parser


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser_regress()

    user_args, user_feature_paths, user_feature_type, fs_subject_dir, \
    meta_data_path, meta_data_format, sample_ids, classes, out_dir, train_perc, \
    num_rep_cv, num_procs, reduced_dim_size, impute_strategy, covar_list, \
    covar_method, grid_search_level, dim_red_method = parse_common_args(
            parser)

    regressor = check_regressor(user_args.regressor)

    # saving the validated and expanded values to disk for later use.
    options_to_save = [sample_ids, classes, out_dir, user_feature_paths,
                       user_feature_type, fs_subject_dir, train_perc, num_rep_cv,
                       reduced_dim_size, num_procs,
                       grid_search_level, regressor, dim_red_method]

    # options_path = save_options(options_to_save, out_dir)
    options_path = None

    return sample_ids, classes, out_dir, options_path, \
           user_feature_paths, user_feature_type, \
           train_perc, num_rep_cv, \
           reduced_dim_size, impute_strategy, num_procs, \
           grid_search_level, regressor, dim_red_method, \
           covar_list, covar_method


def cli():
    """Main entry point, that logs output to stdout as well as a file in out_dir"""

    subjects, classes, out_dir, user_options, user_feature_paths, \
    user_feature_type, train_perc, num_rep_cv, reduced_dim_size, impute_strategy, \
    num_procs, grid_search_level, regressor, dim_red_method, \
    covar_list, covar_method = parse_args()

    print('Running neuropredict version {} for Regression'.format(__version__))

    multi_ds = load_datasets(user_feature_paths, task_type='regress')
    covariates, deconfounder = check_covariates(multi_ds, covar_list, covar_method)

    print(multi_ds)
    impute_strategy = detect_missing_data(multi_ds, impute_strategy)

    scoring = cfg.default_metric_set_regression
    regr_expt = RegressionWorkflow(multi_ds,
                                   pred_model=regressor,
                                   impute_strategy=impute_strategy,
                                   dim_red_method=dim_red_method,
                                   covariates=covariates,
                                   deconfounder=deconfounder,
                                   reduced_dim=reduced_dim_size,
                                   train_perc=train_perc,
                                   num_rep_cv=num_rep_cv,
                                   scoring=scoring,
                                   grid_search_level=grid_search_level,
                                   out_dir=out_dir,
                                   num_procs=num_procs,
                                   user_options=user_options,
                                   checkpointing=True)

    out_results_path = regr_expt.run()
    print('All done.\n')


def make_visualizations():
    pass


class RegressionWorkflow(BaseWorkflow):
    """
    Class defining an neuropredict experiment to be run.

    Encapsulates all the details necessary for execution,
        hence easing the save/load/decide workflow.

    """


    def __init__(self,
                 datasets,
                 pred_model=cfg.default_classifier,
                 impute_strategy=cfg.default_imputation_strategy,
                 covariates=None,
                 deconfounder=cfg.default_deconfounding_method,
                 dim_red_method=cfg.default_dim_red_method,
                 reduced_dim=cfg.default_reduced_dim_size,
                 train_perc=cfg.default_train_perc,
                 num_rep_cv=cfg.default_num_repetitions,
                 scoring=cfg.default_metric_set_regression,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                 out_dir=None,
                 num_procs=cfg.DEFAULT_NUM_PROCS,
                 show_predicted_in_residuals_plot=False,
                 user_options=None,
                 checkpointing=cfg.default_checkpointing):
        super().__init__(datasets,
                         pred_model=pred_model,
                         impute_strategy=impute_strategy,
                         covariates=covariates,
                         deconfounder=deconfounder,
                         dim_red_method=dim_red_method,
                         reduced_dim=reduced_dim,
                         train_perc=train_perc,
                         num_rep_cv=num_rep_cv,
                         scoring=scoring,
                         grid_search_level=grid_search_level,
                         out_dir=out_dir,
                         num_procs=num_procs,
                         user_options=user_options,
                         checkpointing=checkpointing,
                         workflow_type='regress')

        # offering a choice of true vs. predicted target in the residuals plot
        self._show_predicted_in_residuals_plot = show_predicted_in_residuals_plot


    def _eval_predictions(self, pipeline, test_data, true_targets, run_id, ds_id):
        """
        Evaluate predictions and perf estimates to results class.

        Prints a quick summary too, as an indication of progress.
        """

        predicted_targets = pipeline.predict(test_data)
        self.results.add(run_id, ds_id, predicted_targets, true_targets)
        self.results.add_diagnostics(run_id, ds_id, true_targets, predicted_targets)


    def summarize(self):
        """Simple summary of the results produced, for logging and user info"""

        print('\n{}\n'.format(self.results))


    def visualize(self):
        """Method to produce all the relevant visualizations based on the results
        from this workflow."""

        self._compare_metric_distrib()
        self._plot_residuals_vs_target()
        self._plot_feature_importance()


    def _compare_metric_distrib(self):
        """Plot comparing the distributions of different metrics"""

        for metric, m_data in self.results.metric_val.items():
            consolidated = np.empty((self.num_rep_cv, len(m_data)))
            for index, ds_id in enumerate(self.datasets.modality_ids):
                consolidated[:, index] = m_data[ds_id]

            fig_out_path = pjoin(self._fig_out_dir, 'compare_{}'.format(metric))
            compare_distributions(consolidated, self.datasets.modality_ids,
                                  fig_out_path, y_label=metric,
                                  horiz_line_loc=median_of_medians(consolidated),
                                  horiz_line_label='median of medians',
                                  upper_lim_y=None)


    def _plot_residuals_vs_target(self):
        """Important diagnostic plot for regression analyses"""

        if self._show_predicted_in_residuals_plot:
            target_type = 'Predicted targets'
        else:
            target_type = 'True targets'

        target_medians = list()
        residuals, true_targets, predicted = dict(), dict(), dict()
        for index, ds_id in enumerate(self.datasets.modality_ids):
            residuals[ds_id] = self._unroll(self.results.residuals, ds_id)
            predicted[ds_id] = self._unroll(self.results.predicted_targets, ds_id)
            true_targets[ds_id] = self._unroll(self.results.true_targets, ds_id)
            target_medians.append(np.median(true_targets[ds_id]))

        if self._show_predicted_in_residuals_plot:
            targets_to_plot = predicted
        else:
            targets_to_plot = true_targets
        file_suffix = target_type.replace(' ', '_').lower()
        fig_out_path = pjoin(self._fig_out_dir,
                             'residuals_vs_{}'.format(file_suffix))
        multi_scatter_plot(y_data=residuals,
                           x_data=targets_to_plot,
                           fig_out_path=fig_out_path,
                           y_label='Residuals',
                           x_label=target_type,
                           show_zero_line=True, trend_line=None,
                           show_hist=True)

        # variation: predicted vs. target
        fig_out_path = pjoin(self._fig_out_dir, 'predicted_vs_target')
        multi_scatter_plot(y_data=predicted, x_data=true_targets,
                           fig_out_path=fig_out_path,
                           y_label='Predicted target',
                           x_label='True targets',
                           trend_line=np.median(target_medians))


    def _unroll(self, in_dict, ds_id):
        """structure reformat"""

        out_list = list()
        for rep in range(self.num_rep_cv):
            out_list.extend(in_dict[(ds_id, rep)])

        return np.array(out_list)


if __name__ == '__main__':
    cli()
