
__all__ = ['run_workflow', 'rhst', 'visualize', 'freesurfer', 'cli',
           'utils', 'algorithms', 'reports', 'io',
           'config_neuropredict', 'compare', '__version__']

__author__ = 'Pradeep Reddy Raamana, PhD'
__email__  = 'raamana@gmail.com'

from ._version import get_versions
__version__ = get_versions()['version']

# dealing with matplotlib backend
import os
import matplotlib

currently_in_CI = any([os.getenv(var, '').strip().lower() == 'true' for var in ('TRAVIS', 'CONTINUOUS_INTEGRATION')])

def set_agg():
    "set agg as backend"

    matplotlib.use('Agg')
    matplotlib.interactive(False)


if 'DISPLAY' in os.environ:
    display = os.environ['DISPLAY']
    display_name, display_num = display.split(':')
    display_num = int(float(display_num))
    if display_num != 0:
        set_agg()
else:
    set_agg()
    display = None


from sys import version_info
if version_info.major > 2:
    # importing config_neuropredict first (before run_workflow) to avoid a circular situation (when running run_workflow directly)
    from neuropredict import config_neuropredict
    from neuropredict import rhst, visualize, freesurfer, compare, run_workflow
    from neuropredict.run_workflow import cli
    # ^^ importing run_workflow last to  avoid a circular situation (when running run_workflow directly)
else:
    raise NotImplementedError('neuropredict requires Python 3+.')

del get_versions
del version_info