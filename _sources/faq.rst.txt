--------------------------
Frequently Asked Questions
--------------------------

* *What is the overarching goal for neuropredict?*

    * To offer a comprehensive report on predictive analysis effortlessly while following all the best practices!

    * Aiming to interface directly with the outputs of various neuroscience and other popular tools, to reduce the barriers for labs without software expertise and to save time for the labs with software expertise

    * This tool is generic, as it is simply about proper estimation of predictive performance. Hence, there is nothing tied to neuroscience data (despite its name), so users could input arbitrary set of features and targets from any domain (astronomy, nutrition, medicine, phrama or otherwise) and leverage its to produce a comprehensive report.


* *What is your default classification system?*

    * Predictive analysis [by default] is performed with Random Forest classifier/Regressor, after some basic preprocessing comprising of `robust scaling <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html>`_ and `removal of low-variance features <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html>`_.

    * Model selection (grid search of optimal hyper parameters) is performed in an inner cross-validation.


* *Can I use a different classifier?*

    * Yes. User can choose among few techniques offered by ``scikit-learn``, ``xgboost``
    * We plan to support any useful machine learning library as well. `Let me know <http://github.com/raamana/neuropredict/issues/new>`_ if you would like something that's not already integrated. As long as it is implemented in python, we will integrate it.


* *Why did you pick random forests to be the default classifier?*

    * Because they have consistently demonstrated top performance across multiple domains:

        * Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014). Do we Need Hundreds of Classifiers to Solve Real World Classification Problems? Journal of Machine Learning Research, 15, 3133–3181. `[Link] <http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf>`_

        * Lebedev, A. V., Westman, E., Van Westen, G. J. P., Kramberger, M. G., Lundervold, A., Aarsland, D., et al. (2014). Random Forest ensembles for detection and prediction of Alzheimer's disease with a good between-cohort robustness. NeuroImage: Clinical, 6, 115–125. `[Link] <http://doi.org/10.1016/j.nicl.2014.08.023>`_

    * Because it's multi-class by design and automatically estimates feature importance.


* *What are the options for my feature selection?*

  * By default, ``neuropredict`` selects the top ``k = n_train/10`` features based on their variable importance, as computed by Random Forest classifier/regressor, where n_train = number of *training* samples. The value of `n_train` depends on the size of the smallest class in the dataset and is ``train_perc*n_smallest*n_C``, where `train_perc` is the amount of dataset the user reserved for training, `n_C` is the number of classes in the dataset and `n_smallest` is the size of smallest class.
  * you could also choose the value of ``k`` via the ``-k`` / ``--reduced_dim_size`` option

  * The choice of stratifying the training set by the size of smallest class `n_smallest` in the given dataset helps alleviate class-imbalance problems as well as improve the robustness of the classifier.

  * The following dimensionality reduction methods, via the ``-dr`` / ``--dim_red_method`` option, are available at the moment:
    * feature selection: ``SelectKBest_mutual_info_classif``, ``SelectKBest_f_classif``, ``VarianceThreshold``
    * dimensionality reduction: ``Isomap``, ``LLE``, ``LLE_modified``, ``LLE_Hessian``, ``LLE_LTSA``
  * We plan to implement offer even more choices for feature selection and dimensionality reduction in the near future - `Let me know your suggestions <http://github.com/raamana/neuropredict/issues/new>`_. The benefit of trying many arbitrary choices for feature selection method seems unclear. The overarching goals of ``neuropredict`` might help answer the current choice:

    * to enable novice predictive modeling users to get started easily and quickly,

    * provide a thorough estimate of *baseline* performance of their feature sets, instead of trying to find an arbitrary combination of predictive modeling tools to drive the numerical performance as high as possible.

  * Keep in mind, Random forest classifier/regressor automatically discards features without any useful signal.

  * ``neuropredict`` is designed such that another classifier or combination of classifiers could easily be plugged in. We may be adding an option to integrate one of the following options to automatically select a classifier with the highest performance: `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_, `auto_ml <https://github.com/ClimbsRocks/auto_ml>`_ and `tpot <https://github.com/rhiever/tpot>`_ etc.


* *Does neuropredict handle covariates?*

  * **Yes**. In fact, *this is a unique feature for neuropredict*, that is simply not possible in scikit-learn by itself due to some design limitations. We are not aware of any other libraries offering this feature.
  * Using this features requires the use of the `pyradigm data structures <http://raamana.github.io/pyradigm/>`_, which offers you the ability to add in arbitrary set of attributes for each subject.


* *Can I get ROC curves?*

  * Not at the moment, as the presented results and report is obtained from a large number of CV iterations and there is not one ROC curve to represent it all.

  * It is indeed possible to *average* ROC curves from multiple iterations (see below) and visualize it. This feature will be added soon.

    * ROC Reference: Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.

  * For multi-class classification problems, ROC analysis (hyper-surface to be precise) becomes intractable quickly. The author is currently not aware of any easy solutions. if you are aware of any solutions or could contribute, it would be greatly appreciated.


* *Can I compare an arbitrary set of my own custom-designed features?*

  * Yes. That would be quite easy when datasets are managed via ``pyradigm``. You can also use the ``-u`` option to supply arbitrary set of paths where your custom features are stored in a loose / decentralized format (with features for a single samplet stored in a separate folder/file) e.g.

    * ``-y /myproject/awsome-new-idea-v2.0.PyradigmDataset.pkl /project-B/DTI_FA_Method1.PyradigmDataset.pkl``
    * ``-u /sideproject/resting-dynamic-fc /prevproject/resting-dynamic-fc-competingmethod``





