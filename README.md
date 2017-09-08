# neuropredict

neuropredict is part of a broader intiative to develop standardized and easy predictive analysis - see [here](https://drive.google.com/open?id=0BxUb8ldwZEYJR3pCWFpyRUI1YUE) for an overview and the bigger picture idea. 

[![travis](https://travis-ci.org/raamana/neuropredict.svg?branch=master)](https://travis-ci.org/raamana/neuropredict.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/neuropredict.svg)](https://badge.fury.io/py/neuropredict)
[![Python versions](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)]

## Overview
![roleofneuropredict](docs/role.png)


## Goals of the tool

Automatic estimation of predictive power of commonly used neuroimaging features as well as user-defined features.

The aim of this python module would be to automatically assess the predictive power of commonly used neuroimaging features (such as resting-state connectivity, fractional anisotropy, subcortical volumes and cortical thickness features) automatically read from the processing of popular tools such as FSL, DTIstudio, AFNI and Freesurfer, and present a comprehensive report on a given dataset. It is mainly aimed (to lower or remove the barriers) at clinical users who would like to understand what features and brain regions are discriminative in their shiny new dataset before diving into the deep grey sea of feature extraction and optimization.

neuropredict sounds similar (on the surface) to other software available, such as scikit-learn (which is used underneath), however our aim here is to lower the barriers as much as possible, or remove them altogether and make machine learning seamless! For example, 
 * You don't have to code when you use neurpredict - saves you a lot of headache with development and debugging!
 * You don't have to learn how the machine learning toolkits work and their APIs. This can be daunting and time-consuming, and  can likely lead to bad choices made in terms of how proper cross-validation is done.
 * Toolkits like scikit-learn are geared towards developers (focusing on the API to support most generic uses), but not focused easing your workflow, esp. when analyzing many feature sets. 
 * Comprehensive analysis of misclassfied subjects (histogram at top right in figure below) in different pairs of classes is not in the interest of other toolkits and not possible without significant rewriting of many underlying components of scikit-learn.
 * Thorough analysis of  misclassification rates for different feature sets (radar plot at bottom right in figure below) is not possible in scikit-learn without significant development (who only provide basic metrics of classifier performance).. 
 * Methods and tools for statistical comparison of multiple features (and models) is missing in other toolkits, and which is a priority for neuropredict.

All you would need to provide are commonly used features (such as a Freesurfer output directory) and obtain an easy to read report (see below), along with well-packaged export of performance metrics (for sharing and posthoc comparison) on the predictive power of the features they are interested in.

It is primary geared towards neuroscience data (where the need for machine learning is high) offering readers for popular tools. However, it is not restricted to neuro-data, you could input any arbitrary set of features (from astrophysics, biology or chemistry).

**Happy machine learning!**

And neuropredicting.

![composite](docs/composite_flyer.001.png)

**Table of Contents**

- [neuropredict](#)
	- [FAQ](#faq)
	- [Context](#context)
	- [Predictive analysis](#predictive-analysis)
	- [Report](#report)
- [Input Features](#input-features)
	- [Arbitray feature input](#arbitray-feature-input)
	- [Automatic readers currently supported](#automatic-readers-currently-supported)
	- [Automatic readers in development (stay tuned)](#automatic-readers-in-development-stay-tuned)
- [Installation:](#installation)
- [Usage:](#usage)
- [Dependencies](#dependencies)

## FAQ

Refer to ![FAQ](FAQ.md)

## Context

Imagine you have just acquired a wonderful new dataset with certain number of diseased patients and healthy controls. In the case of T1 mri analysis, you typically start by preprocessing it wih your favourite software (such as Freesurfer), which produces a ton of segmentations and statistics within them (such as their volumes and cortical thickness). Typical scenario would be to examine group differences (e.g. between controls and disease_one or between controls and other_disease), find the most discriminative variables and/or their brain regions and report how they relate to know cognitive or neuropsychological measures. This analysis and the resulting insights is necessary and informs us better of the dataset. However, that's not the fullest extent of the analysis one could perform, as association studies do not inform us of the predictive utility of the aforementioned discriminative variables or regions, which needs to independently investigated.

## Predictive analysis
 Conducting a machine learning study (to assess the predictive utility of different regions, features or methods) is not trivial. In the simplest case, it requires one to understand standard techniques, learn one or two toolboxes and do the complex programming necessary to interface their data with ML toolbox (even with the help of well-written packages like nilearn that are meant for neuroimaging analysis). In addition, in order to properly evaluate the performance, the user needs to have a good grasp of the best practices in machine learning. Even if the user could produce certain numbers out of a black-box toolboxes, some more programming is necessary to make sense of the results and procude necessary plots for publications.

## Report
 Neuropredict is here to remove those barriers and make your life easier!

 All you need to do is take care of preprocessing and produce quality controlled output through popular software, and neuropredict will produce a comprehensive report (see figures below) of distribtions of cross-validated performance, confusion matrices, analysis into misclassification and an intuitive comparison across multiple features.

## Example 
  For example, if you have a dataset with the following three classes: 5 controls, 6 disease_one and 9 other_disease, all you would need to do is produce a meta data file as shown below (specifying a class label for each subject):

```
3071,controls
3069,controls
3064,controls
3063,controls
3057,controls
5004,disease_one
5074,disease_one
5077,disease_one
5001,disease_one
5002,disease_one
5003,disease_one
5000,other_disease
5006,other_disease
5013,other_disease
5014,other_disease
5016,other_disease
5018,other_disease
5019,other_disease
5021,other_disease
5022,other_disease
```


and `neuropredict` will produce the figures (and numbers in a CSV files) as shown here:

![composite](docs/composite_flyer.001.png)

The higher resolution PDFs are included in the [docs](docs) folder.

I hope this user-friendly tool would help you get started on the predictive analysis you've been wanting to do for a while.

# Input Features

neuropredict is aimed at interfacing with popular feature extraction algorithms such as Freesurfer, FSL and others directly - see *Readers* section below. However, it allows an arbitray input of features that have already been extracted via user's own pipeline(s). 

## Arbitray feature input
For custom input: 
 * the user needs to save the features in a single folder for all subjects (let's call it /project/myawsomepipeline )
  * specify it with --userdefined /project/myawsomepipeline
 * within which, features for each subject in a separate folder (named after its id specified in the meta data file)
 * in a file called `features.txt`. The `features.txt` file must contain a single floating point number per line (see below - its not comma separated), 
 * and all the subject features must have an equal number of features. 
 
Then neuropredict will automatically consolidate the features into its native [`pyradigm` MLdataset format](github.com/raamana/pyradigm), ideally suited for the predictive analysis tasks.

The example for a dataset with 2 controls and 2 disease sujects with 5 features each is shown below: 
```
$ 11:19:22 linux userdefined >>  ls -1
control-001
control-002
disease-003
disease-004
$ 11:19:30 linux userdefined >>  tree
.
|-- control-001
|   `-- features.txt
|-- control-002
|   `-- features.txt
|-- disease-003
|   `-- features.txt
`-- disease-004
    `-- features.txt

4 directories, 4 files
$ 11:19:33 linux userdefined >>  head -n 5 */features.txt
==> control-001/features.txt <==
0.868896136902
0.542305564899
0.115903893374
0.503297862357
0.564961631104

==> control-002/features.txt <==
0.868896136902
0.542305564899
0.115903893374
0.503297862357
0.564961631104

==> disease-003/features.txt <==
0.868896136902
0.542305564899
0.115903893374
0.503297862357
0.564961631104

==> disease-004/features.txt <==
0.868896136902
0.542305564899
0.115903893374
0.503297862357
0.564961631104
```

## Automatic readers currently supported
* Freesurfer
  * Subcortical volumes
  * Wholebrain Aseg stats
  
## Automatic readers in development (stay tuned)
* Freesurfer
  * cortical thickness
  * gray matter density
  * structural covariance
* Any nibabel-readable data
* DT-MRI features
* task-free fMRI features
* HCP datasets
* Weka's ARFF format

# Installation

neuropredict could be easily installed by issuing the following command:
```bash
pip install neuropredict
```

If `pip` throws an error, re-run the above command few times, most errors usually get resolved.

Installing it with admin privileges is the recommended way. However, if you do not have admin privileges, try this:
```bash
pip install neuropredict --user
```

However, you may need to add the location of binary files to your path by adding this command to your login script:
```bash
export PATH=$PATH:~/.local/bin/
```

## version comptability

 - Neuropredict is continuously tested to work on Python 3.6, 3.5 and 2.7. 
 - It's not guaranteed to work for the remaining versions.

# Usage:

```
usage: neuropredict [-h] -m METADATAFILE -o OUTDIR [-f FSDIR]
                    [-u USER_FEATURE_PATHS [USER_FEATURE_PATHS ...] | -d
                    DATA_MATRIX_PATH [DATA_MATRIX_PATH ...]]
                    [-p POSITIVECLASS] [-t TRAIN_PERC] [-n NUM_REP_CV]
                    [-a ATLASID] [-s [SUBGROUP [SUBGROUP ...]]]

optional arguments:
  -h, --help            show this help message and exit
  -m METADATAFILE, --metadatafile METADATAFILE
                        Abs path to file containing metadata for subjects to
                        be included for analysis. At the minimum, each subject
                        should have an id per row followed by the class it
                        belongs to. E.g. sub001,control sub002,control
                        sub003,disease sub004,disease
  -o OUTDIR, --outdir OUTDIR
                        Output folder to store features and results.
  -f FSDIR, --fsdir FSDIR
                        Absolute path to SUBJECTS_DIR containing the finished
                        runs of Freesurfer parcellation (each subject named
                        after its ID in the metadata file). E.g. --fsdir
                        /project/freesurfer_v5.3
  -u USER_FEATURE_PATHS [USER_FEATURE_PATHS ...], --user_feature_paths USER_FEATURE_PATHS [USER_FEATURE_PATHS ...]
                        List of absolute paths to an user's own features. Each
                        folder contains a separate folder for each subject
                        (named after its ID in the metadata file) containing a
                        file called features.txt with one number per line. All
                        the subjects (in a given folder) must have the number
                        of features (#lines in file). Different folders can
                        have different number of features for each subject.
                        Names of each folder is used to annotate the results
                        in visualizations. Hence name them uniquely and
                        meaningfully, keeping in mind these figures will be
                        included in your papers. E.g. --user_feature_paths
                        /project/fmri/ /project/dti/ /project/t1_volumes/ .
                        Only one of user_feature_paths and user_feature_paths
                        options can be specified.
  -d DATA_MATRIX_PATH [DATA_MATRIX_PATH ...], --data_matrix_path DATA_MATRIX_PATH [DATA_MATRIX_PATH ...]
                        List of absolute paths to text files containing one
                        matrix of size N x p (num_samples x num_features).
                        Each row in the data matrix file must represent data
                        corresponding to sample in the same row of the meta
                        data file (meta data file and data matrix must be in
                        row-wise correspondence). Name of this file will be
                        used to annotate the results and visualizations. E.g.
                        --data_matrix_path /project/fmri.csv /project/dti.csv
                        /project/t1_volumes.csv. Only one of
                        user_feature_paths and user_feature_paths options can
                        be specified.File format could be 1) a simple comma-
                        separated text file (with extension .csv or .txt):
                        which can easily be read back with
                        numpy.loadtxt(filepath, delimiter=',') or 2) a numpy
                        array saved to disk (with extension .npy or .numpy)
                        that can read in with numpy.load(filepath). One could
                        use numpy.savetxt(data_array, delimiter=',') or
                        numpy.save(data_array) to save features.File format is
                        inferred from its extension.
  -p POSITIVECLASS, --positiveclass POSITIVECLASS
                        Name of the positive class (Alzheimers, MCI or
                        Parkinsons etc) to be used in calculation of area
                        under the ROC curve. Applicable only for binary
                        classification experiments. Default: class appearning
                        second in order specified in metadata file.
  -t TRAIN_PERC, --trainperc TRAIN_PERC
                        Percentage of the smallest class to be reserved for
                        training. Must be in the interval [0.01 0.99].If
                        sample size is sufficiently big, we recommend 0.5.If
                        sample size is small, or class imbalance is high,
                        choose 0.8.
  -n NUM_REP_CV, --numrep NUM_REP_CV
                        Number of repetitions of the repeated-holdout cross-
                        validation. The larger the number, the better the
                        estimates will be.
  -a ATLASID, --atlas ATLASID
                        Name of the atlas to use for visualization. Default:
                        fsaverage, if available.
  -s [SUBGROUP [SUBGROUP ...]], --subgroup [SUBGROUP [SUBGROUP ...]]
                        This option allows the user to study different
                        combinations of classes in multi-class (N>2) dataset.
                        For example, in a dataset with 3 classes CN, FTD and
                        AD, two studies of pair-wise combinations can be
                        studied with the following flag --subgroup CN,FTD
                        CN,AD . This allows the user to focus on few
                        interesting subgroups depending on their dataset/goal.
                        Format: each subgroup must be a comma-separated list
                        of classes. Hence it is strongly recommended to use
                        class names without any spaces, commas, hyphens and
                        special characters, and ideally just alphanumeric
                        characters separated by underscores. Default: all -
                        using all the available classes in a all-vs-all multi-
                        class setting. Beware: this feature has not been tested heavily.
```

# Dependencies
 * numpy
 * scikit-learn
 * pyradigm
 * nibabel
 * scipy
 * matplotlib

## Support on Beerpay
Hey dude! Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/raamana/neuropredict/badge.svg?style=beer-square)](https://beerpay.io/raamana/neuropredict)  [![Beerpay](https://beerpay.io/raamana/neuropredict/make-wish.svg?style=flat-square)](https://beerpay.io/raamana/neuropredict?focus=wish)
