
__all__ = ['run_workflow', 'rhst', 'visualize', 'freesurfer', 'cli',
           'config_neuropredict', 'model_comparison', '__version__']

__author__ = 'Pradeep Reddy Raamana, PhD'
__email__  = 'raamana@gmail.com'

from ._version import get_versions
__version__ = get_versions()['version']

from sys import version_info
if version_info.major > 2:
    # importing config_neuropredict first (before run_workflow) to avoid a circular situation (when running run_workflow directly)
    from neuropredict import config_neuropredict
    from neuropredict import rhst, visualize, freesurfer, model_comparison, run_workflow
    from neuropredict.run_workflow import cli
    # ^^ importing run_workflow last to  avoid a circular situation (when running run_workflow directly)
else:
    raise NotImplementedError('neuropredict requires Python 3+.')

del get_versions
del version_info