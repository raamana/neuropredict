Implemenation details
---------------------

In line with the overraching goals of neuropredict, the following choices were made:

 - cross-validation scheme (central to performance estimation) has been chosen to be repeated hold-out (referred to as ShuffleSplit in scikit-learn). As you may be aware, choice of CV scheme makes a difference, and this is a good choice among available options. RepeatedKFold is also possible, at the expense slightly more awkward implementation


You may also be interested in :doc:`faq`.