
import os
import numpy as np

def aseg_stats_whole_brain(fspath, subjid):
    """

    Returns a feature set of volumes found in Freesurfer output: subid/stats/aseg.stats


    """

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


def aseg_stats_subcortical(fspath, subjid):
    """
    Returns all the subcortical volumes found in stats/aseg.stats.

    Equivalent of load_fs_segstats.m

    """

    ignore_seg_names = ['WM-hypointensities', 'Left-WM-hypointensities', 'Right-WM-hypointensities',
                        'non-WM-hypointensities', 'Left-non-WM-hypointensities', 'Right-non-WM-hypointensities',
                        'Optic-Chiasm']

    segstatsfile = os.path.join(fspath, subjid, 'stats', 'aseg.stats')

    # ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
    stats = np.loadtxt(segstatsfile, dtype="i1,i1,i4,f4,S32,f4,f4,f4,f4,f4")

    filtered_stats = [ (seg[1], seg[3], seg[4]) for seg in stats if seg[4] not in ignore_seg_names ]

    seg_ids, volumes, names = zip(*filtered_stats)

    return np.array(volumes) # , np.array(seg_ids), list(names)


def fsthickness(path, subjid, fwhm=10):
    """
    Returns thickness feature set at a given fwhm.

    Assumes freesurfer was run with -qcache flag!

    """



if __name__ == '__main__':
    pass