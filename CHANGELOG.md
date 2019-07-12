# Change Log

## 0.5.2

 - Imputation of missing values
 - Additional classifiers such as `XGBoost`, Decision Trees
 - Better internal code structure
 - Lot more tests
 - More precise tests, as we vary number of classes wildly in test suites
 - several bug fixes and enhancements (more cmd line options such as `--print_options`)

## 0.4.5

 - new classifier : SVM
 - new flag to choose a feature selection method
 - user chosen options now saved to disk, to better handle complex interactions between options
 - code clean up and faster tests

## 0.4.1

 - Parallelizing the main the CV loop, leading to great reduction in total time for report generation!
 - More options, including choice of different classifiers (Random Forest and Extra Trees classifiers)
 - Support for dataset in Weka's ARFF format
 - Better visualizations (handling small/nan values in feature importance, layouts and design)
 - auto versioning!
 - Ability to read meta data from pyradigms or ARFF files, without having to specify that separately.
 - Dropping support for Python 2.7 :(

## 0.3.1
