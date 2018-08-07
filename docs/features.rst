
Input formats
-------------

Currently supported:

 * Arbitrary/user-defined format (see below)
 * CSV files (with samples along rows and features along columns)
 * `pyradigm`'s `MLDataset <http://pyradigm.readthedocs.io>`_
 * Weka's `ARFF <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_


Formats specific to neuroimaging:

 * Freesurfer

  * Subcortical volumes
  * Wholebrain Aseg stats


In development (stay tuned)

 * Freesurfer

   * cortical thickness
   * gray matter density
   * structural covariance
 * Any nibabel-readable data
 * DT-MRI features
 * task-free fMRI features
 * HCP datasets
 * Weka's ARFF format

Interfaces to Neuroimaging tools
--------------------------------

neuropredict is aimed at interfacing with popular feature extraction algorithms such as Freesurfer, FSL and others directly - see *Readers* section below. However, it allows an arbitrary input of features that have already been extracted via user's own pipeline(s).

Arbitrary feature input
-------------------------

For custom input:

* the user needs to save the features in a single folder for all subjects (let's call it ``/project/myawsomepipeline`` )
    * specify it with --userdefined /project/myawsomepipeline
* within which, features for each subject in a separate folder (named after its id specified in the meta data file)
* in a file called ``features.txt``. The ``features.txt`` file must contain a single floating point number per line (see below - its not comma separated),
* and all the subject features must have an equal number of features.

Then neuropredict will automatically consolidate the features into its native `pyradigm MLdataset format <github.com/raamana/pyradigm>`_, ideally suited for the predictive analysis tasks.

The example for a dataset with 2 controls and 2 disease sujects with 5 features each is shown below:

.. code-block:: bash

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
