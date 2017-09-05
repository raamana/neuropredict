
__all__ = ['neuropredict', 'rhst', 'visualize', 'freesurfer',
           'config_neuropredict', 'model_comparison']

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    import neuropredict, config_neuropredict, rhst, visualize, freesurfer, model_comparison
elif version_info.major > 2:
    from neuropredict import neuropredict, config_neuropredict, rhst, visualize, freesurfer, model_comparison
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')

