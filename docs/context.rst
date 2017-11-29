Context
-------

Imagine you have just acquired a wonderful new dataset with certain number of diseased patients and healthy controls. In the case of T1 mri analysis, you typically start by preprocessing it wih your favourite software (such as Freesurfer), which produces a ton of segmentations and statistics within them (such as their volumes and cortical thickness). Typical scenario would be to examine group differences (e.g. between controls and disease_one or between controls and other_disease), find the most discriminative variables and/or their brain regions and report how they relate to know cognitive or neuropsychological measures. This analysis and the resulting insights is necessary and informs us better of the dataset. However, that's not the fullest extent of the analysis one could perform, as association studies do not inform us of the predictive utility of the aforementioned discriminative variables or regions, which needs to independently investigated.

Predictive analysis
-------------------

Conducting a machine learning study (to assess the predictive utility of different regions, features or methods) is not trivial. In the simplest case, it requires one to understand standard techniques, learn one or two toolboxes and do the complex programming necessary to interface their data with ML toolbox (even with the help of well-written packages like nilearn that are meant for neuroimaging analysis). In addition, in order to properly evaluate the performance, the user needs to have a good grasp of the best practices in machine learning. Even if the user could produce certain numbers out of a black-box toolboxes, some more programming is necessary to make sense of the results and procude necessary plots for publications.

And, that's the complex burden neuropredict is designed to eliminate.

All you would need to provide are:

 - your features for each sample and
 - which classes these samples belong to,

 and neuropredict

 - produces an easy to read report (see below),
 - along with well-packaged export of performance metrics (for sharing and posthoc comparison)

.. image:: role.png
