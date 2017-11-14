-------
Report
-------

Neuropredict is here to remove those barriers and make your life easier!

All you need to do is take care of preprocessing and produce quality controlled output through popular software, and neuropredict will produce a comprehensive report (see figures below) of distribtions of cross-validated performance, confusion matrices, analysis into misclassification and an intuitive comparison across multiple features.

.. image:: composite_flyer.001.png


Results Interpretation
----------------------

We will walk you through the resulting visualizations one by one.


Comparison of predictive accuracy
-------------------------------------

The following visualization compares the predictive performance of four features, using the balanced accuracy metric. The performance of each feature is shown as a distribution, wherein multiple points in the distribution are obtained from multiple CV repetitions. As you can see,

 - it is important to visualize the full distribution and not just the mean or median, as the distributions are typically wide and usually not normal.
 - to test whether a particular feature is statistically significantly better, a distribution for each feature is necessary to run statistical tests.

.. image:: results_interpretation.balanced_accuracy.pdf


Comparison of misclassification rates
-------------------------------------

.. image:: results_interpretation.mcr_radar.png


**under dev**

*Stay tuned*