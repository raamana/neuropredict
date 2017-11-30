--------------------------------------------------------------------------------------------------
About
--------------------------------------------------------------------------------------------------

neuropredict : automatic estimation of predictive power of commonly used neuroimaging features as well as user-defined features.


The aim of neuropredict is to automatically assess the predictive power of different sets of features (such as resting-state connectivity, fractional anisotropy, subcortical volumes and cortical thickness features) provided by the user, or automatically read from the processing of popular tools such as FSL, DTIstudio, AFNI and Freesurfer. It is aimed (to lower or remove the barriers) at clinical users interested in measuring the classification accuracy of different features they are interested in studying, and assess  importance of brain regions/features in their dataset.


On a high level,

.. image:: high_level_flow.png


On a more detailed level,

.. image:: role.png



``neuropredict`` sounds similar (on the surface) to other software available, such as scikit-learn (which is used underneath), however our aim here is to lower the barriers as much as possible, or remove them altogether and make machine learning seamless! For example,

 * You don't have to code when you use neurpredict - saves you a lot of headache with development and debugging!
 * You don't have to learn how the machine learning toolkits work and their APIs. This can be daunting and time-consuming, and  can likely lead to bad choices made in terms of how proper cross-validation is done.
 * Toolkits like scikit-learn and nilearn are geared towards developers (focusing on the API to support most generic uses), but not focused easing your workflow, esp. when analyzing many feature sets.
 * Comprehensive analysis of misclassfied subjects (histogram at top right in figure below) in different pairs of classes is not in the interest of other toolkits and not possible without significant rewriting of many underlying components of scikit-learn.
 * Thorough analysis of  misclassification rates for different feature sets (radar plot at bottom right in figure below) is not possible in scikit-learn without significant development (who only provide basic metrics of classifier performance)..
 * Methods and tools for statistical comparison of multiple features (and models) is missing in other toolkits, and which is a priority for neuropredict.

All you would need to provide are your own features (such as a Freesurfer output directory) and obtain an easy to read and comprehensive report on the predictive power of the features they are interested in, along with well-packaged export of performance metrics (for sharing and posthoc comparison).

It is geared towards neuroscience data (where the need for machine learning is high) offering readers for popular tools. However, it is not restricted to neuro-data, you could input any arbitrary set of features (from astrophysics, biology or chemistry).

**Happy machine learning!**

And neuropredicting.

Check the :doc:`usage_cli` and :doc:`features` pages, and let me know your comments.

Thanks for checking out. Your feedback will be appreciated.