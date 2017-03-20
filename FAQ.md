# neuropredict FAQ

## Frequently Asked Questions

1. What is the overarching goal for neuropredict?
  * To offer a comprehensive report on predictive analysis effortlessly!
  * Aiming to interface directly with the outputs of various neuroimaging tools
   - although the user could input arbitrary set of features (neuroimaging, or otherwise).
2. What is your classification system?
  * Predictive analysis is performed with Random Forest classifier (using scikit-learn's implementation) 
  * Model selection (grid search of optimal hyper parameters) is performed in the inner cross-validation
  * No feature selection is performed currently, as the random forest has an internal mechanism of discarding uninformative features! 
   * However this feature may be supported depending on the user's demand.
3.    
   
