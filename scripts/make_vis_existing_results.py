
from neuropredict.run_workflow import make_visualizations as make_vis
from os.path import join as pjoin, exists as pexists, abspath, realpath, dirname, basename

wdir = '/u1/work/hpc3194/4RTNI/processed/graynet/freesurfer_curv_GLASSER2016_fwhm10_range-0.3_0.3_nbins25/predict_comparison'

res_path = pjoin(wdir, 'rhst_results.pkl')
make_vis(res_path, wdir)