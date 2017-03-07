
import neuropredict
import freesurfer

fsdir = '/u1/work/hpc3194/PPMI/processed/freesurfer'
meta  = '/Users/Reddy/rotman/psy/ppmi_meta.csv'

volumes, indices, names = freesurfer.aseg_stats_subcortical(fsdir, '3617_bl_PPMI')

volumes