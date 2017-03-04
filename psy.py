#/usr/bin/python

import sys
import os
import nibabel
import sklearn
import argparse
from pyradigm import MLDataset

def not_unspecified( var ):
    """ Checks for null values of a give variable! """

    return var not in [ 'None', None, '' ]


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = argparse.ArgumentParser(prog="psy")

    parser.add_argument("-m", "--metadatafile", action="store", dest="metadatafile",
                        default=None, required=True,
                        help="Abs path to file containing metadata for subjects to be included for analysis. At the "
                             "minimum, each subject should have an id per row followed by the class it belongs to. "
                             "E.g. \n"
                             "sub001,control\n"
                             "sub002,control\n"
                             "sub003,disease\n"
                             "sub004,disease\n")

    parser.add_argument("-f", "--fsdir", action="store", dest="fsdir",
                        default=None,
                        help="Abs. path of SUBJECTS_DIR containing the finished runs of Freesurfer parcellation")

    parser.add_argument("-u", "--userdir", action="store", dest="userdir",
                        default=None,
                        help="Abs. path to an user's own features."
                             "This contains a separate folder for each subject (named after its ID in the metadata "
                             "file) containing a file called features.txt with one number per line. All the subjects "
                             "must have the number of features (#lines in file)")

    # TODO perhaps I can have two arguments: one to specify feature type (which determines the reader), and another
    # to obtain the folder path to read from.

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

    metadatafile = os.path.abspath(options.metadatafile)
    assert os.path.exists(metadatafile), "Given metadata file doesn't exist."

    if not_unspecified(options.fsdir):
        fsdir = os.path.abspath(options.fsdir)
        assert os.path.exists(fsdir), "Given Freesurfer directory doesn't exist."
        userdir = None
    elif not_unspecified(options.userdir):
        fsdir = None
        userdir = os.path.abspath(options.userdir)
        assert os.path.exists(userdir), "Given user-defined directory for features doesn't exist."
    else:
        raise IOError('One of Freesurfer or user-defined directory must be specified.')

    outdir = os.path.abspath(options.outdir)
    if not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except:
            raise

    return metadatafile, outdir, userdir, fsdir


def getmetadata(path):
    """Populates the dataset dictionary with subject ids and classes"""


def fsvolumes(path, subjid):
    """Returns a feature set of volumes found in aparc+aseg.stats"""


def fsthickness(path, subjid, fwhm=10):
    """
    Returns thickness feature set at a given fwhm.

    Assumes freesurfer was run with -qcache flag!

    """


def userdefinedget(featdir, subjid):
    """
    Method to read in features for a given subject from a user-defined feature folder. This featdir must contain a
    separate folder for each subject with a file called features.txt with one number per line.

    :param featdir:
    :return: vector of numbers.
    """

    featfile = os.path.join(featdir, subjid, 'features.txt')
    with open(featfile,'r') as fid:
        data = fid.read().splitlines()

    return data


def getfeatures(subjects, classes, featdir, getmethod = None):
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
    for subjid in subjects:
        data = getmethod(featdir, subjid)
        ds.add_sample(subjid, data, class_labels[classes[subjid]], classes[subjid])

    # save the dataset to disk to enable passing on multiple dataset(s)
    savepath = os.path.join(featdir, 'features_{}.MLDataset'.format(getmethod.__name__))
    ds.save(savepath)

    return ds


def run_rhst(datasets, outdir):
    """



    :param datasets: dictionary of MLdataset features
    :param outdir: output folder to save the results.
    :return:

    """



def run():
    """Main entry point."""

    metadatafile, outdir, userdir, fsdir = parse_args()

    subjects, classes = getmetadata(metadatafile)

    # let's start with one method/feature set for now
    if not_unspecified(userdir):
        path_to_features = userdir
        chosenmethod = userdefinedget
    else:
        path_to_features = fsdir
        chosenmethod = fsvolumes

    # # this could be a list of methods when RHsT is able to handle it.
    # chosenmethod = [ fsvolumes, fsthickness, userdefinedget ]

    dataset = getfeatures(subjects, classes, path_to_features, getmethod = chosenmethod)

    run_rhst(dataset, outdir)



if __name__ == '__main__':
    run()