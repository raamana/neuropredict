
__all__ = ['run_workflow', 'rhst', 'visualize', 'freesurfer',
           'config_neuropredict', 'model_comparison']

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    import config_neuropredict, rhst, run_workflow, visualize, freesurfer, model_comparison
elif version_info.major > 2:
    # importing config_neuropredict first (before run_workflow) to avoid a circular situation (when running run_workflow directly)
    from neuropredict import config_neuropredict
    from neuropredict import rhst, visualize, freesurfer, model_comparison, run_workflow
    # ^^ importing run_workflow last to  avoid a circular situation (when running run_workflow directly)
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')

