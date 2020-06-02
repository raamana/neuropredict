
Usage
-----------------------

.. note::

    Since version 0.6, to leverage the advanced features of handling confounds/covariates within nested cross-validation, ``neuropredict`` requires the input datasets in the ``pyradigm`` format. This is needed to accurately infer the names and data-types of confounding variables and values, which is very hard or impossible with in CSV files, esp when dealing multiple modalities/feature-sets. Learn more about this data structure at http://raamana.github.io/pyradigm/.

    ``pyradigm`` datasets not only enable great use of neuropredict, but also help with reproducibility in a number of ways, including making it easier to share the datasets between collaborators and colleagues etc, but also to track their provenance with various user-defined attributes. More @ http://raamana.github.io/pyradigm/.


The command line interface for neuropredict is strongly recommended (given its focus on batch processing multiple comparisons). There are two main interfaces to neuropredict: the :doc:`usage_clf_cli` and the :doc:`usage_regr_cli`. Check their respective pages for instructions on their usage.

The high-level differences between the two workflows are the following:

 - The targets for prediction in the classification workflow are categorical (i.e. health vs. disease, monkey vs. chair etc), whereas in the regression workflow they are continuous (and numerical).
 - In classification workflow, you can have more than two classes (from now on referred to as targets to be consistent), and hence offer the ability to select a sub-group (subset of classes) for analysis. For example, if your dataset has 4 classes A, B, C, and D, you can choose to analyze one binary comparison A vs. B, and a 3-class comparison B vs. C vs. D with the following flag ``-sg A,B B,C,D``. That concept of sub-grouping does not exist in the regression workflows
 - In addition, the concepts of class imbalance, and stratifying the training set to control the class sizes in the training and/or test sets exist only in the classification workflow.
 - The performance metrics change drastically between the two workflows: classification analyses often focus on accuracy, AUC and confusion matrices, whereas regression analyses discuss ``r2``, ``MAE``, explained variance and ``MSE`` etc. Hence, the results saved to disk differ in their structure at the metric level (as well as in additional attributes), and hence needed to be handled separately depending on the workflow.
 -


