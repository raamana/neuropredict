# psy

Automatic estimation of predictive power of commonly used structural imaging features as well as user-defined features.

The aim of this python module would be to automatically assess the predictive power of commonly used structural imaging features (such as subcortical volumes, cortical thickness features) from Freesurfer, and present a comprehensive report on a given dataset. It is mainly aimed (to lower or remove the barriers) at clinical users who would like to understand what features and brain regions are discriminative in their shiny new dataset before diving into the deep grey sea of feature extraction and optimization.

PS: It sounds similar to nilearn on the surface, however it is aimed to lower the barriers even further, or remove them altogether! All the user would need to provide are commonly used features (such as a Freesurfer output directory) and they get a easy to read/publish report on the predictive power of the features they are interested in.

## usage:

```
usage: psy [-h] -m METADATAFILE [-f FSDIR] [-u USERDIR] [-o OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  -m METADATAFILE, --metadatafile METADATAFILE
                        Abs path to file containing metadata for subjects to
                        be included for analysis. At the minimum, each subject
                        should have an id per row followed by the class it
                        belongs to. E.g. sub001,control sub002,control
                        sub003,disease sub004,disease
  -f FSDIR, --fsdir FSDIR
                        Abs. path of SUBJECTS_DIR containing the finished runs
                        of Freesurfer parcellation
  -u USERDIR, --userdir USERDIR
                        Abs. path to an user's own features.This contains a
                        separate folder for each subject (named after its ID
                        in the metadata file) containing a file called
                        features.txt with one number per line. All the
                        subjects must have the number of features (#lines in
                        file)
  -o OUTDIR, --outdir OUTDIR
                        Output folder to store features and results.
```




