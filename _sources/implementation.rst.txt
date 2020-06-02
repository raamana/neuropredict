Implemenation details
---------------------

In line with the overarching goals of neuropredict, the following choices were made:

 - cross-validation scheme (central to performance estimation) has been chosen to be repeated hold-out (referred to as ShuffleSplit in scikit-learn). As you may be aware, choice of CV scheme makes a difference, and this is a good choice among available options. RepeatedKFold is also possible, at the expense of slightly more awkward implementation, as well as making subsequent statistics (such as significance tests) less tractable.


You may also be interested in :doc:`faq`.