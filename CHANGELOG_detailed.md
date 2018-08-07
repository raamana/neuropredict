Changelog
=========


0.4.1 (2017-11-06)
------------------
- Untracking not so useful script. [Pradeep Reddy Raamana]
- Test for arff usage. [Pradeep Reddy Raamana]
- Improving font sizes in plots. [Pradeep Reddy Raamana]
- Adding support for Weka's ARFF input datasets. [Pradeep Reddy Raamana]
- Updating the flyer. [Pradeep Reddy Raamana]
- Rest syntax fixes. [Pradeep Reddy Raamana]
- Updating FAQ to reflect the latest implementation. [Pradeep Reddy
  Raamana]
- Improvign the tests for visualization. [Pradeep Reddy Raamana]
- Better handling of importance values for features never tested.
  [Pradeep Reddy Raamana]
- Option to change classifier implemented!! [Pradeep Reddy Raamana]
- Correcting stupid mistake. [Pradeep Reddy Raamana]
- Minor bug fix in misclf freq calc. [Pradeep Reddy Raamana]
- Getting the check of display check working on CI. [Pradeep Reddy
  Raamana]
- Better corr with xtick in metric compare plot. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Better checking of whether DISPLAY is available. [Pradeep Reddy
  Raamana]
- Numbering the labels for accuracy plot for easy ref when n is large.
  [Pradeep Reddy Raamana]
- Feat imp plots now handle never-selected features, display 95% CI
  only. [Pradeep Reddy Raamana]
- Using non-interactive mode for pyplot only when DISPLAY is not set.
  [Pradeep Reddy Raamana]
- Minor improvements to plots; version info to output logs. [Pradeep
  Reddy Raamana]
- Using a different backend Agg for headless vis. [Pradeep Reddy
  Raamana]
- Improving accuracy comparison plots, when legends are long; darker
  colors. [Pradeep Reddy Raamana]
- Throwing in extra trees clf -- need to integrate and test. [Pradeep
  Reddy Raamana]
- Fixing presentation of chance accuracy: training classes are
  stratified! [Pradeep Reddy Raamana]
- Trying to implement API interface. [Pradeep Reddy Raamana]
- New option to make vis from existing results [skip ci] [Pradeep Reddy
  Raamana]
- Added the -version feature [skip ci] [Pradeep Reddy Raamana]
- Logging datasets info common to all datasets for reference. [Pradeep
  Reddy Raamana]
- Trying to disable parallelism within sklearn withOUT setting n_jobs;
  Option for NO grid search at all. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Refactoring exhaustive search to be multiple levels; exposing in CLI.
  [Pradeep Reddy Raamana]
- Add code of conduct [skip ci] [Pradeep Reddy Raamana]
- Better docs. [Pradeep Reddy Raamana]
- With pyradigms as input, no need to specify meta data separately;
  output folder is optional, writes results to $PWD/neuropredict.
  [Pradeep Reddy Raamana]
- Link correction. [Pradeep Reddy Raamana]
- Better docs. [Pradeep Reddy Raamana]
- Makefile from cookiecutter. [Pradeep Reddy Raamana]
- Not optimizing for min_impurity_decrease unless proven needed [skip
  ci] [Pradeep Reddy Raamana]
- Falling back to sequential eval when #procs == 1. [Pradeep Reddy
  Raamana]
- Much improved help text with raw formatting [skip ci] [Pradeep Reddy
  Raamana]
- Choosing 50 reps for testing. [Pradeep Reddy Raamana]
- More robust query of available cpu count. [Pradeep Reddy Raamana]
- Preprocessing with robust scaling; option less exhaustive grid search;
  log flush. [Pradeep Reddy Raamana]
- Setting RF param grid within RHsT itself. [Pradeep Reddy Raamana]
- Boarding grid search scope for RF; more options for feat sel size.
  [Pradeep Reddy Raamana]
- Merge branch 'parallel_rhst' [Pradeep Reddy Raamana]
- Clean up unused; rename run_cli to cli. [Pradeep Reddy Raamana]
- Merge pull request #16 from raamana/parallel_rhst. [Pradeep Reddy
  Raamana]

  Parallelizing CV repetitions
- Adding auto versioning via versioneer (#15) [Kevin Le]
- Finishing the parameter spec for the API [skip ci] [Pradeep Reddy
  Raamana]
- Auto versioning. [Pradeep Reddy Raamana]
- Update CONTRIBUTING.md. [Pradeep Reddy Raamana]


0.3.1 (2017-09-29)
------------------
- Dropping support for python 2.7 :( [Pradeep Reddy Raamana]
- Improving the chance accuracy tests. [Pradeep Reddy Raamana]
- Finalizing option to choose a subgroup to process. [Pradeep Reddy
  Raamana]
- Implement subgroups option. [Pradeep Reddy Raamana]
- Option to choose a subgroup to process [skip ci] [Pradeep Reddy
  Raamana]
- Fixing chance accuracy calculations and annotating it. [Pradeep Reddy
  Raamana]
- New flag to specify number of CPUs [skip ci] [Pradeep Reddy Raamana]
- Better gathering of feature importances [skip ci] [Pradeep Reddy
  Raamana]
- Returning & gathering misclassification frequencies explicitly [skip
  ci] [Pradeep Reddy Raamana]
- Rm unnecessary files [skip ci] [Pradeep Reddy Raamana]
- Calculation of chance accuracy based on class imbalance [skip ci]
  [Pradeep Reddy Raamana]
- Basic skeleton to gather results from parallel CV runs [skip ci]
  [Pradeep Reddy Raamana]
- Tiny script generate visualizations from existing results. [Pradeep
  Reddy Raamana]
- Basic skeletion for exporting first before visualization [ skip ci]
  [Pradeep Reddy Raamana]
- - parallelizing main loop over repetitions w/ shared data: to test
  [skip ci] - also saving results from each iteration independently -
  ignoring some warnings known to be bugs. [Pradeep Reddy Raamana]
- To be tested init for parallel implementation. [Pradeep Reddy Raamana]
- Adapting the new shape of confuison matrix container. [Pradeep Reddy
  Raamana]
- Major refactoring of rhst module to streamline it via smaller methods.
  [Pradeep Reddy Raamana]
- Parallelizign the main repetition loop in rhst; reformat code;
  [Pradeep Reddy Raamana]
- Matching n_jobs with predispatch in gridsearch! [Pradeep Reddy
  Raamana]
- Converting all asserts to if not raise. [Pradeep Reddy Raamana]
- Parallel grid search with 4 cpus; minor changes to output and notes;
  [Pradeep Reddy Raamana]
- Ignoring NaNs in computing medians, better outputs. [Pradeep Reddy
  Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Making the test dataset really small to speed up testing. [Pradeep
  Reddy Raamana]
- Fixing # values to unpack in test. [Pradeep Reddy Raamana]
- Reducing grid search time with smaller grid, efficient feat selectors
  [skip ci] [Pradeep Reddy Raamana]
- Brute forcing module discovery: installing in devel mode on CI.
  [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Theme override try from mac. [Pradeep Reddy Raamana]
- Adding static html files for gh-pages [skip ci] [Pradeep Reddy
  Raamana]
- Getting sklearn pipeline and gridsearch combo to work. [Pradeep Reddy
  Raamana]
- No argparse table wrap [skip ci] [Pradeep Reddy Raamana]
- More docs [skip ci] [Pradeep Reddy Raamana]
- Minor bug fixes and code cleanup. [Pradeep Reddy Raamana]
- Falling back to test rudimentary RF and local grid search [skip ci]
  [Pradeep Reddy Raamana]
- Ensuring each subgroup has atleast two unique classes; better
  docs;[skip ci] [Pradeep Reddy Raamana]
- Better docs. [Pradeep Reddy Raamana]
- Removing dead code [skip ci] [Pradeep Reddy Raamana]
- Better reorg of input handling [skip ci] [Pradeep Reddy Raamana]
- Fixing mistaken deletion [skip ci] [Pradeep Reddy Raamana]
- Initial implementation of enabling generic selection of sklearn
  pipelines [skip ci] [Pradeep Reddy Raamana]
- Treating the best params as a dict to enable use of models other than
  RF [skip ci] [Pradeep Reddy Raamana]
- Finalizing grid search implementation [skip ci] [Pradeep Reddy
  Raamana]
- Additional checks on feat sel size calc [skip ci] [Pradeep Reddy
  Raamana]
- New method refactoring instantiation of pipeline & param grid  [skip
  ci] [Pradeep Reddy Raamana]
- Initial implementation of Pipeline mechanism: MI top K + RF; [skip ci]
  changes to dim red size calculation to make it slightly larger;
  [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Changing default parameters and steps a bit to speedup [skip ci]
  [Pradeep Reddy Raamana]
- Trying out GridSearchCV for optimization of random forests - to be
  tested [skip ci] [Pradeep Reddy Raamana]
- Quick skeleton for API access; better API docs [skip ci] [Pradeep
  Reddy Raamana]
- Better non-occluded legends for cobweb misclf plots [skip ci] [Pradeep
  Reddy Raamana]
- Closing figures as they are done exporting [skip ci] [Pradeep Reddy
  Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Set theme jekyll-theme-dinky. [Pradeep Reddy Raamana]
- Ensuring the combined name doesnt get longer than allowed [skip ci]
  [Pradeep Reddy Raamana]
- Improved docs [skip ci] [Pradeep Reddy Raamana]
- Ensuring 1/num_classes appears on metric distribution for better
  presentation [skip ci] [Pradeep Reddy Raamana]
- Test setup fixes for 2.7. [Pradeep Reddy Raamana]
- New option to set size for feature selection; code/docs clean up.
  [Pradeep Reddy Raamana]
- Module neuropredict removed, renamed to run_workflow.py. [Pradeep
  Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]

  # Conflicts:
  #	neuropredict/__main__.py
- Few renamings. [Pradeep Reddy Raamana]
- Showing importance for only the top 25 features; python 3 support
  [skip ci] [Pradeep Reddy Raamana]
- Major pkg reord to minimize confusion and circular imports; python 3
  support tweaking. [Pradeep Reddy Raamana]
- Option to input pyradigm as input; code cleanup. [Pradeep Reddy
  Raamana]
- Better org for subgroup parsing. [Pradeep Reddy Raamana]
- Not messing with css overrides - no text wrapping for argparse [skip
  ci] [Pradeep Reddy Raamana]
- Adding the missed css file. [Pradeep Reddy Raamana]
- Requirements file for docs [skip ci] [Pradeep Reddy Raamana]
- Docs organized [skip ci] [Pradeep Reddy Raamana]
- First try at docs [skip ci] [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Adding version badges [skip ci] [Pradeep Reddy Raamana]
- Badges! [skip ci] [Pradeep Reddy Raamana]
- Version info [skip ci] [Pradeep Reddy Raamana]
- Separating parser for docs; shorthand for path handlers. [Pradeep
  Reddy Raamana]
- Fewer samples, lower dim with more CV reps! parallelizing travis
  builds; [Pradeep Reddy Raamana]
- Increasing timeout for CI and reducing dimensionality for faster CV
  runs. [Pradeep Reddy Raamana]
- Omitting python 3.2-3.4. [Pradeep Reddy Raamana]
- Trusting pytest/travis for import mechanism of 3.6 over PyCharm.
  [Pradeep Reddy Raamana]
- Tests succeed in python 2.7/3.6 according to PyCharm - lets see about
  CI. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Better description referring to subplots [skip ci] [Pradeep Reddy
  Raamana]
- Markdown typo [skip ci] [Pradeep Reddy Raamana]
- Answering why neuropredict [skip ci] [Pradeep Reddy Raamana]
- Fixing a minor bug in warning message construction [skip ci] [Pradeep
  Reddy Raamana]
- First steps towards python 3 compatibility; cleanup imports [skip ci]
  [Pradeep Reddy Raamana]
- Confusion matrix title now reflects the feature set name. [Pradeep
  Reddy Raamana]
- Updating FAQ to match the latest version. [Pradeep Reddy Raamana]
- Adding link to large ML comparison study. [Pradeep Reddy Raamana]
- Setting up CI. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Correcting dependencies [skip ci] [Pradeep Reddy Raamana]
- Update usage. [Pradeep Reddy Raamana]
- Add Beerpay's badge. [Pradeep Reddy Raamana]
- Update CONTRIBUTING.md. [Pradeep Reddy Raamana]
- Update CONTRIBUTING.md. [Pradeep Reddy Raamana]
- Adding the graphical slide on role and where it fits in the predictive
  analysis workflow. [Pradeep Reddy Raamana]
- Improving the setup and requirements. [Pradeep Reddy Raamana]
- Example datasets. [Pradeep Reddy Raamana]
- Csv reading bug fix; adding setuptools to requirements. [Pradeep Reddy
  Raamana]
- - bug fix: when features.txt contains a single number, get method
  returns a 0-d scalar for which len() is not defined. Fixed it to
  return an array always. - feature addition: new visualization module
  to generate bar plots for comparison of misclassification rates for
  binary experiments - better handling of exceptions always throwing
  tracebacks - updated max_dimensionality_to_avoid_curseofdimensionality
  to not request more features than available. [Pradeep Reddy Raamana]
- Adding slide on where np fits in the predictive analysis workflow.
  [Pradeep Reddy Raamana]
- Adding link to overview slides. [Pradeep Reddy Raamana]
- - turning the exporting and reading in of results via a dictionary and
  not sure if this would solve the problem of having a fixed variable
  order. [Pradeep Reddy Raamana]
- Create CONTRIBUTING.md. [Pradeep Reddy Raamana]
- Italicizing questions for clarity. [Pradeep Reddy Raamana]
- Delete freesurfer_features.py. [Pradeep Reddy Raamana]
- - new parallel coordinate plot for 2-class misclf rate visualization
  to better align with multi-class cobweb plot. [Pradeep Reddy Raamana]
- - bug fix: when features.txt contains a single number, get method
  returns a 0-d scalar for which len() is not defined. Fixed it to
  return an array always. - feature addition: new visualization module
  to generate bar plots for comparison of misclassification rates for
  binary experiments - better handling of exceptions: always throwing
  tracebacks - updated max_dimensionality_to_avoid_curseofdimensionality
  to not request more features than available. [Pradeep Reddy Raamana]
- Removing build and egg-info from version tracking. [Pradeep Reddy
  Raamana]
- - additional assertion to ensure there are no duplicate sample ids in
  dataset - minor other fixes. [Pradeep Reddy Raamana]
- - feature addition: option to specify multiple subgroups in a N>2
  multi-class experiment. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- FAQ arbitrary set of custom features. [Pradeep Reddy Raamana]
- - feature addition: support for numpy file format .npy as a data
  matrix. [Pradeep Reddy Raamana]
- - feature addition: implementation of reader for a data matrix in
  file. [Pradeep Reddy Raamana]
- Removing empty old config file. [Pradeep Reddy Raamana]
- - separating the method misclassification percentage calculation -
  also saving the list of subjects that were selected for testing -
  including them for exporting results - removing unneeded parameters
  and cleanup. [Pradeep Reddy Raamana]
- - feature addition: allowing an arbitrary combination of multiple
  user-defined features and Freesurfer features. [Pradeep Reddy Raamana]
- - Comment on ROC curves. [Pradeep Reddy Raamana]
- - fixing the clipping of metric distribution at the bottom. [Pradeep
  Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- FAQ on covariates. [Pradeep Reddy Raamana]
- - user prompt even if features for 1 subject could not be read -
  checking for the availability of x-server - more try/except to make
  the context more precise - correct name for scikit learn in
  requirements. [Pradeep Reddy Raamana]
- - fixing the entry point script - updating the readme with install
  troubleshooting. [Pradeep Reddy Raamana]
- - removing console script option in setup.py - slight cleanup - saving
  positive class specified to disk. [Pradeep Reddy Raamana]
- Add install instructions and reorg. [Pradeep Reddy Raamana]
- New file created as a reminder to implement nemenyi test. [Pradeep
  Reddy Raamana]
- More generic and efificient code for minor things. [Pradeep Reddy
  Raamana]
- Cleanup. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Update FAQ.md. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Update FAQ.md. [Pradeep Reddy Raamana]
- New test to ensure chance accuracy on random datasets. [Pradeep Reddy
  Raamana]
- Better reorg of the main entry module. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Create FAQ.md. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Initial trial for unit test for chance classifier. [Pradeep Reddy
  Raamana]
- Fixing the export of confusion matrix. [Pradeep Reddy Raamana]
- - new module to export the different results to per-dataset CSVs -
  saving method names along with the results. [Pradeep Reddy Raamana]
- - feature names are integrated into the work flow (esp. feat
  imporatance) - freesurfer feature extraction methods return feature
  names/labels - tidying up the feature imp plot to horizontal bar plot.
  [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Flag names update. [Pradeep Reddy Raamana]
- Improved misclassification histogram plots. [Pradeep Reddy Raamana]
- Cleaner/more accurate names for functions. [Pradeep Reddy Raamana]
- - rough implementation of test for random dataset test. [Pradeep Reddy
  Raamana]
- - choosing the get method correctly for custom user input - cosmetics.
  [Pradeep Reddy Raamana]
- Adding table of contents. [Pradeep Reddy Raamana]
- Documentation for custom input features. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Composite flyer for promotion. [Pradeep Reddy Raamana]
- Fixing and organizing colormaps. [Pradeep Reddy Raamana]
- Update README.md. [Pradeep Reddy Raamana]
- Output images to be included in README. [Pradeep Reddy Raamana]
- Updated README. [Pradeep Reddy Raamana]
- Merge remote-tracking branch 'origin/master' [Pradeep Reddy Raamana]
- Ignoring headerlines in the metadata file. [Pradeep Reddy Raamana]
- - two new flags to allow the user to choose training perc and num reps
  - new method to ensure reloaded dataset matches the meta data spec -
  implementation and integration of radar plots for comparison of pair-
  wise misclassification rates. [Pradeep Reddy Raamana]
- - positive class flag to enable AUC calculation - minor changes
  elsewhere. [Pradeep Reddy Raamana]
- - new reader for aseg stats whole brain using numpy.fromregex - minor
  changes elsewhere. [Pradeep Reddy Raamana]
- Minor changes. [Pradeep Reddy Raamana]
- - implementation & integration of subcortical aseg stats - captured
  all the constant values to a config file. [Pradeep Reddy Raamana]
- - implementation & integration of module to summarize misclassified
  subjects - warning if features for too many subjects cant be read.
  [Pradeep Reddy Raamana]
- - Implemented and integrated the variable importance plots - Removed
  fixed seeding of random forest to make it truly random. [Pradeep Reddy
  Raamana]
- - Fixing a miscalculation in conf mat calculation and improving the
  presentation. [Pradeep Reddy Raamana]
- - implemented and integrated the accuracy  and confusion matrix
  visualization - Tried to preserve label order in CM across different
  RHsT runs [TODO test] [Pradeep Reddy Raamana]
- Reorg of the structure for packaging implementation of whole-brain seg
  volume reading basic integration test with fsvolumes reading and RHsT.
  [Pradeep Reddy Raamana]
- Packaging setup and registration to PyPi. [Pradeep Reddy Raamana]
- Renaming to neuropredict, improving the README Implementing Counters
  for misclassified sample ids. [Pradeep Reddy Raamana]
- Preliminary implementation of RHsT and RF are done basic integration
  tests between RHsT and RF are done methods to save and load results
  done. [Pradeep Reddy Raamana]
- Almost complete implementation of random forest optimization routine.
  TODO need to ensure class order is maintained over multiple trials of
  RHsT. [Pradeep Reddy Raamana]
- Improving the flow in RHsT. [Pradeep Reddy Raamana]
- Initial implementation of RHsT. [Pradeep Reddy Raamana]
- - Enabled methods to allow input of user-defined features. - Updated
  the readme with better usage notes. [Pradeep Reddy Raamana]
- Initial barebones skeleton. [Pradeep Reddy Raamana]
- Initial commit. [Pradeep Reddy Raamana]


