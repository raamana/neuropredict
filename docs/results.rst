-------
Report
-------

neuropredict produces a comprehensive report, parts of which can be seen in the figure below:

.. image:: composite_outputs.png
  :width: 700


The report consists of

 - distribtions of cross-validated performance (balanced accuracy),
 - confusion matrices for each feature set,
 - intuitive comparison of misclassification rates across multiple features.

The output directory (chosen with ``-o`` flag) contains the following sets of files, whose names are self-explanatory. In each set, there is separate visualization (PDF) or CSV file produced for each feature (named features A, B, C and D below) being studied, for your convenience.

**First**, a set of visualizations in PDF format:

.. parsed-literal ::

  balanced_accuracy.pdf
  compare_misclf_rates.pdf
  confusion_matrix_FeatureA.pdf
  confusion_matrix_FeatureB.pdf
  confusion_matrix_FeatureC.pdf
  confusion_matrix_FeatureD.pdf
  feature_importance.pdf
  misclassified_subjects_frequency_histogram.pdf

**Second**, a set of CSV files in a subfolder called ``exported_results`` (typical contents shown below), which can used for further posthoc statisical analysis or to produce more customised visualizations.

.. parsed-literal ::

  average_misclassification_rates_FeatureA.csv
  average_misclassification_rates_FeatureB.csv
  average_misclassification_rates_FeatureC.csv
  average_misclassification_rates_FeatureD.csv
  balanced_accuracy.csv
  confusion_matrix_FeatureA.csv
  confusion_matrix_FeatureB.csv
  confusion_matrix_FeatureC.csv
  confusion_matrix_FeatureD.csv
  feature_importance_FeatureA.csv
  feature_importance_FeatureB.csv
  feature_importance_FeatureC.csv
  feature_importance_FeatureD.csv
  subject_misclf_freq_FeatureA.csv
  subject_misclf_freq_FeatureB.csv
  subject_misclf_freq_FeatureC.csv
  subject_misclf_freq_FeatureD.csv

**Third**, a Python pickle file ``rhst_results.pkl`` containing the full set of results, that neuropredict can bases the visualizatiosn on.

**Fourth**, a set of files named ``misclassified_subjects_*_most_frequent.txt``. Each of these files (for each feature being studied) contain the set of subject IDs that were misclassified different features most frequently (during multiple repetitions of CV). This is to help you further dig into why they have been so often misclassified. If classes are expected to well separated, and some subjects misclassified consistently (over 60%), you may want to ask these questions:

 - are they mislabelled (data entry or human error)?
 - were their data quality controlled properly (preprocessing or QC failure)?
 - what explains their 'outlier' status (demographics, neuropsych or something related)?

Fifth, few miscellaneous set of files that neuropredict relies to produce the comprehensive report.

Interpretation
----------------------

We will walk you through the resulting visualizations one by one.


Comparison of predictive accuracy
-------------------------------------

The following visualization compares the predictive performance of four features, using the balanced accuracy metric. The performance of each feature is shown as a distribution, wherein multiple points in the distribution are obtained from multiple CV repetitions. As you can see,

 - it is important to visualize the full distribution and not just the mean or median, as the distributions are typically wide and usually not normal.
 - to test whether a particular feature is statistically significantly better, a distribution for each feature is necessary to run statistical tests.

.. image:: results_interpretation.balanced_accuracy.png
  :width: 700


Comparison of misclassification rates
-------------------------------------

.. image:: results_interpretation.mcr_radar.png
  :width: 700


**These docs will be futher improved soon. Stay tuned!**


