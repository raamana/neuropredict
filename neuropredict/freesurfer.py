
import os
import numpy as np

def fsvolumes(fspath, subjid):
    """

    Returns a feature set of volumes found in Freesurfer output: subid/stats/aseg.stats


    """
    # TODO need to include subcortical volumes

    # 'eTIV' is not included as it is used to norm subcortical volumes
    seg_names_sel = [ 'BrainSegVol', 'BrainSegVolNotVent',
        'lhCortexVol', 'rhCortexVol', 
        'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol', 
        'SubCortGrayVol', 'TotalGrayVol', 
        'SupraTentorialVol', 'SupraTentorialVolNotVent', 
        'MaskVol', 'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV', 
        'lhSurfaceHoles', 'rhSurfaceHoles' ]

    wb_seg_volumes_selected = np.full([len(seg_names_sel), 1], np.nan)

    num_header_lines = 13
    num_measures = 21  # len(wb_seg_names_selected)

    segstatsfile = os.path.join(fspath, subjid, 'stats', 'aseg.stats')

    seg_names = [None]*num_measures
    seg_volumes = np.full([num_measures, 1], np.nan)

    with open(segstatsfile) as ssf:
        for _ in range(num_header_lines):
            ssf.readline()

        for ix in range(num_measures):
            line = ssf.readline()
            parts = line.strip().split(',')
            seg_names[ix] = parts[1].strip()
            seg_volumes[ix] = np.float64(parts[3].strip())

    selected_names_volumes = [ (vol[0], name) for (vol, name) in zip(seg_volumes, seg_names) if name in seg_names_sel]
    sel_volumes, sel_names = zip(*selected_names_volumes)

    return np.array(sel_volumes) # sel_names not returned to follow return-a-single-vector convention


def subcortical_aseg_stats(path, subjid):
    """
    Returns all the subcortical volumes found in stats/aseg.stats.

    Equivalent of load_fs_segstats.m

    """
    # TODO reader for subcortical volumes: equivalent of load_fs_segstats.m

    subctx_volumes = None

    return np.array(subctx_volumes)

def fsthickness(path, subjid, fwhm=10):
    """
    Returns thickness feature set at a given fwhm.

    Assumes freesurfer was run with -qcache flag!

    """



if __name__ == '__main__':
    pass