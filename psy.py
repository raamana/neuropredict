import sys
import os
import nibabel
import sklearn
import argparse
from pyradigm import MLDataset

def parse_args():
    """Parser/validator for the cmd line args."""

    parser = argparse.ArgumentParser(prog="psy")

    parser.add_argument("-f", "--fsdir", action="store", dest="fsdir",
                        default=None,
                        help="Abs. path of SUBJECTS_DIR containing the finished runs of Freesurfer parcellation")

    parser.add_argument("-m", "--metadatafile", action="store", dest="metadatafile",
                        default=None,
                        help="Abs path to file containing metadata for subjects to be included for analysis. At the "
                             "minimum, each subject should have an id per row followed by the class it belongs to. "
                             "E.g. "
                             "sub001,control"
                             "sub002,control"
                             "sub003,disease"
                             "sub004,disease")

    parser.add_argument("-o", "--outdir", action="store", dest="outdir",
                        default=None,
                        help="Output folder to store features and results.")

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        options = parser.parse_args()
    except:
        parser.exit(1)

    fsdir = os.path.abspath(options.fsdir)
    metadatafile = os.path.abspath(options.metadatafile)
    outdir = os.path.abspath(options.outdir)

    assert os.path.exists(fsdir), "Given Freesurfer directory doesnt exist."
    assert os.path.exists(metadatafile), "Given metadata file doesnt exist."
    if not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except:
            raise

    return fsdir, metadatafile, outdir


def getmetadata(path):
    """Populates the dataset dictionary with subject ids and classes"""


def fsvolumes(path):
    """Returns a feature set of volumes found in aparc+aseg.stats"""


def fsthickness(path, fwhm=10):
    """
    Returns thickness feature set at a given fwhm.

    Assumes freesurfer was run with -qcache flag!

    """


def getfeatures(subjects, classes, getmethod = None):
    """Populates the pyradigm data structure with features from a given method.

    getmethod: takes in a path and returns a vectorized feature set (e.g. set of subcortical volumes).
    classes: dict of class labels keyed in by subject id

    """

    assert callable(getmethod), "Supplied getmethod is not callable!" \
                                "It must take in a path and return a vectorized feature set."

    # generating an unique numeric label for each class (sorted in order of their appearance in metadata file)
    class_set = set(classes.values())
    class_labels = list()
    for idx, cls in enumerate(class_set):
        class_labels[cls] = idx

    ds = MLDataset()
    for sub in subjects:
        data = getmethod(sub)
        ds.add_sample(sub, data, class_labels[classes[sub]], classes[sub])

    return ds


def run_rhst(datasets, outdir):
    """



    :param datasets: dictionary of MLdataset features
    :param outdir: output folder to save the results.
    :return:

    """



def run():
    """Main entry point."""

    fsdir, metadatafile, outdir = parse_args()

    subjects = getmetadata(metadatafile)

    # let's start with one method/featureset for now
    dataset = getfeatures(subjects, getmethod = 'fsvolumes')

    run_rhst(dataset, outdir)



if __name__ == '__main__':
    run()