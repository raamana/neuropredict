
Command line interface
-----------------------

The command line interface for graynet (preferred interface, given its target is HPC) is shown below. Check the bottom of this page for examples.

.. argparse::
   :module: neuropredict.neuropredict
   :func: get_parser
   :prog: neuropredict
   :nodefault:
   :nodefaultconst:

A rough example of usage can be:

.. code-block:: bash

    neuropredict -m meta_data.csv -f /work/project/features_dir


**Example for meta-data**

  For example, if you have a dataset with the following three classes: 5 controls, 6 disease_one and 9 other_disease, all you would need to do is produce a meta data file as shown below (specifying a class label for each subject):

.. parsed-literal::

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


and `neuropredict` will produce the figures (and numbers in a CSV files) as shown here:

.. image:: composite_flyer.001.png

The higher resolution PDFs are included in the `docs <docs/results_vis>`_ folder.

I hope this user-friendly tool would help you get started on the predictive analysis you've been wanting to do for a while.


