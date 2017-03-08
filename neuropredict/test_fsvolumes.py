
import neuropredict
import freesurfer

fsdir = '/u1/work/hpc3194/PPMI/processed/freesurfer'
meta  = '/Users/Reddy/rotman/psy/ppmi_meta.csv'

volumes_regex = freesurfer.aseg_stats_whole_brain_via_regex(fsdir, '3617_bl_PPMI')

volumes = freesurfer.aseg_stats_whole_brain(fsdir, '3617_bl_PPMI')

volumes