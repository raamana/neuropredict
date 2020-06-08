Implementation details
----------------------

In line with the overarching goals of neuropredict, the following choices were made:

 - cross-validation scheme (central to performance estimation) has been chosen to be repeated hold-out (referred to as ``ShuffleSplit`` in scikit-learn). As you may be aware, choice of CV scheme makes a difference, and this is a good choice among available options. Check my tutorial on CV to learn why `repeated holdout CV is a safe choice <https://crossinvalidation.com/2020/06/04/unambiguous-terminology-for-data-splits-in-nested-cross-validation-cv-training-tuning-and-reporting-sets/>`_.
 - RepeatedKFold is also possible, at the expense of slightly more awkward implementation, as well as making subsequent statistics (such as significance tests etc) less tractable. Hence, we are not offering this CV scheme at the moment.

You may also be interested in :doc:`faq`.