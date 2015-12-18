#!/usr/bin/env python
import sys

sys.path.append('../.')
sys.path.append('.')

import os
import os.path as path
import getpass
import glob
import optparse
import socket
import time
import numpy as np
import h5py
import itertools
import vigra
import copy
import pgmlink as track
from empryonic import io
import json
from trackingfeatures import get_feature_vector
from progressbar import ProgressBar

def getConfigAndCommandLineArguments():

    usage = """%prog [options] FILES
Track cells.

Before processing, input files are copied to OUT_DIR. Groups, that will not be modified are not
copied but linked to the original files to improve execution speed and storage requirements.
"""

    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--config-file', type='string', dest='config', default=None,
                      help='path to config file')
    parser.add_option('--method', type='str', default='conservation',
                      help='conservation or conservation-dynprog [default: %default]')
    parser.add_option('-o', '--output-dir', type='str', dest='out_dir', default='tracked', help='[default: %default]')
    parser.add_option('--x-scale', type='float', dest='x_scale', default=1., help='[default: %default]')
    parser.add_option('--y-scale', type='float', dest='y_scale', default=1., help='[default: %default]')
    parser.add_option('--z-scale', type='float', dest='z_scale', default=1., help='[default: %default]')
    parser.add_option('--comment', type='str', dest='comment', default='none',
                      help='some comment to log [default: %default]')
    parser.add_option('--random-forest', type='string', dest='rf_fn', default=None,
                      help='use cellness prediction instead of indicator function for (mis-)detection energy')
    parser.add_option('-f', '--forbidden_cost', type='float', dest='forbidden_cost', default=0,
                      help='forbidden cost [default: %default]')
    parser.add_option('--min-ts', type='int', dest='mints', default=0, help='[default: %default]')
    parser.add_option('--max-ts', type='int', dest='maxts', default=-1, help='[default: %default]')
    parser.add_option('--min-size', type='int', dest='minsize', default=0,
                      help='minimal size of objects to be tracked [default: %default]')
    parser.add_option('--dump-traxelstore', type='string', dest='dump_traxelstore', default=None,
                      help='dump traxelstore to file [default: %default]')
    parser.add_option('--load-traxelstore', type='string', dest='load_traxelstore', default=None,
                      help='load traxelstore from file [default: %default]')
    parser.add_option('--raw-data-file', type='string', dest='raw_filename', default='',
                      help='filename to the raw h5 file')
    parser.add_option('--raw-data-path', type='string', dest='raw_path', default='volume/data',
                      help='Path inside the raw h5 file to the data')
    parser.add_option('--dump-hypotheses-graph', type='string', dest='hypotheses_graph_filename', default=None,
                      help='save hypotheses graph so it can be loaded later')

    parser.add_option('--json-output', type='string', dest='json_filename', default=None,
                      help='filename where to save the generated JSON file to')


    consopts = optparse.OptionGroup(parser, "conservation tracking")
    consopts.add_option('--max-number-objects', dest='max_num_objects', type='float', default=2,
                        help='Give maximum number of objects one connected component may consist of [default: %default]')
    consopts.add_option('--max-neighbor-distance', dest='mnd', type='float', default=30, help='[default: %default]')
    consopts.add_option('--max-nearest-neighbors', dest='max_nearest_neighbors', type='int', default=1, help='[default: %default]')
    consopts.add_option('--division-threshold', dest='division_threshold', type='float', default=0.1, help='[default: %default]')
    # detection_rf_filename in general parser options
    consopts.add_option('--size-dependent-detection-prob', dest='size_dependent_detection_prob', action='store_true')
    # forbidden_cost in general parser options
    # ep_gap in general parser options
    consopts.add_option('--average-obj-size', dest='avg_obj_size', type='float', default=0, help='[default: %default]')
    consopts.add_option('--without-tracklets', dest='without_tracklets', action='store_true')
    consopts.add_option('--with-opt-correct', dest='woptical', action='store_true')
    consopts.add_option('--det', dest='detection_weight', type='float', default=10.0, help='detection weight [default: %default]')
    consopts.add_option('--div', dest='division_weight', type='float', default=10.0, help='division weight [default: %default]')
    consopts.add_option('--dis', dest='disappearance_cost', type='float', default=500.0, help='disappearance cost [default: %default]')
    consopts.add_option('--app', dest='appearance_cost', type='float', default=500.0, help='appearance cost [default: %default]')
    consopts.add_option('--tr', dest='transition_weight', type='float', default=10.0, help='transition weight [default: %default]')
    consopts.add_option('--motionModelWeight', dest='motionModelWeight', type='float', default=0.0, help='motion model weight [default: %default]')
    consopts.add_option('--without-divisions', dest='without_divisions', action='store_true')
    consopts.add_option('--means', dest='means', type='float', default=0.0,
                        help='means for detection [default: %default]')
    consopts.add_option('--sigma', dest='sigma', type='float', default=0.0,
                        help='sigma for detection [default: %default]')
    consopts.add_option('--with-merger-resolution', dest='with_merger_resolution', action='store_true', default=False)
    consopts.add_option('--without-constraints', dest='woconstr', action='store_true', default=False)
    consopts.add_option('--trans-par', dest='trans_par', type='float', default=5.0,
                        help='alpha for the transition prior [default: %default]')
    consopts.add_option('--border-width', dest='border_width', type='float', default=0.0,
                        help='absolute border margin in which the appearance/disappearance costs are linearly decreased [default: %default]')
    consopts.add_option('--ext-probs', dest='ext_probs', type='string', default=None,
                        help='provide a path to hdf5 files containing detection probabilities [default:%default]')
    consopts.add_option('--objCountPath', dest='obj_count_path', type='string',
                        default='/CellClassification/Probabilities/0/',
                        help='internal hdf5 path to object count probabilities [default=%default]')
    consopts.add_option('--divPath', dest='div_prob_path', type='string', default='/DivisionDetection/Probabilities/0/',
                        help='internal hdf5 path to division probabilities [default=%default]')
    consopts.add_option('--featsPath', dest='feats_path', type='string',
                        default='/ObjectExtraction/RegionFeatures/0/[[%d], [%d]]/Default features/%s',
                        help='internal hdf5 path to object features [default=%default]')
    consopts.add_option('--translationPath', dest='trans_vector_path', type='str',
                        default='OpticalTranslation/TranslationVectors/0/data',
                        help='internal hdf5 path to translation vectors [default=%default]')
    consopts.add_option('--labelImgPath', dest='label_img_path', type='str',
                        default='/ObjectExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]',
                        help='internal hdf5 path to label image [default=%default]')
    consopts.add_option('--timeout', dest='timeout', type='float', default=1e+75, help='CPLEX timeout in sec. [default: %default]')
    consopts.add_option('--with-graph-labeling', dest='w_labeling', action='store_true', default=False, 
                        help='load ground truth labeling into hypotheses graph for further evaluation on C++ side,\
                        requires gt-path to point to the groundtruth files')

    parser.add_option_group(consopts)

    optcfg, args = parser.parse_args()

    if(optcfg.config != None):
        with open(optcfg.config) as f:
            configfilecommands = f.read().splitlines()
        optcfg, args2 = parser.parse_args(configfilecommands)

    numArgs = len(args)
    if numArgs == 0 or optcfg.json_filename == None:
        parser.print_help()
        sys.exit(1)

    return optcfg, args

def generate_traxelstore(h5file,
                         options,
                         feature_path,
                         time_range,
                         x_range,
                         y_range,
                         z_range,
                         size_range,
                         x_scale=1.0,
                         y_scale=1.0,
                         z_scale=1.0,
                         with_div=True,
                         with_local_centers=False,
                         median_object_size=None,
                         max_traxel_id_at=None,
                         with_merger_prior=True,
                         max_num_mergers=1,
                         with_optical_correction=False,
                         ext_probs=None
):
    print "generating traxels"
    print "filling traxelstore"
    ts = track.TraxelStore()
    fs = track.FeatureStore()
    max_traxel_id_at = track.VectorOfInt()

    print "fetching region features and division probabilities"
    print h5file.filename, feature_path

    detection_probabilities = []
    division_probabilities = []

    if with_div:
        print options.div_prob_path
        divProbs = h5file[options.div_prob_path]

    if with_merger_prior:
        detProbs = h5file[options.obj_count_path]

    if with_local_centers:
        localCenters = None  # self.RegionLocalCenters(time_range).wait()

    if x_range is None:
        x_range = [0, sys.maxint]

    if y_range is None:
        y_range = [0, sys.maxint]

    if z_range is None:
        z_range = [0, sys.maxint]

    shape_t = len(h5file[options.obj_count_path].keys())
    keys_sorted = range(shape_t)

    if time_range is not None:
        if time_range[1] == -1:
            time_range[1] = shape_t
        keys_sorted = [key for key in keys_sorted if time_range[0] <= int(key) < time_range[1]]
    else:
        time_range = (0, shape_t)

    filtered_labels = {}
    obj_sizes = []
    total_count = 0
    empty_frame = False

    for t in keys_sorted:
        feats_name = options.feats_path % (t, t + 1, 'RegionCenter')
        #region_centers = np.array(feats[t]['0']['RegionCenter'])
        region_centers = np.array(h5file[feats_name])

        feats_name = options.feats_path % (t, t + 1, 'Coord<Minimum>')
        lower = np.array(h5file[feats_name])
        feats_name = options.feats_path % (t, t + 1, 'Coord<Maximum>')
        upper = np.array(h5file[feats_name])

        if region_centers.size:
            region_centers = region_centers[1:, ...]
            lower = lower[1:, ...]
            upper = upper[1:, ...]
        if with_optical_correction:
            try:
                feats_name = options.feats_path % (t, t + 1, 'RegionCenter_corr')
                region_centers_corr = np.array(h5file[feats_name])
            except:
                raise Exception, 'cannot consider optical correction since it has not been computed'
            if region_centers_corr.size:
                region_centers_corr = region_centers_corr[1:, ...]

        feats_name = options.feats_path % (t, t + 1, 'Count')
        #pixel_count = np.array(feats[t]['0']['Count'])
        pixel_count = np.array(h5file[feats_name])
        if pixel_count.size:
            pixel_count = pixel_count[1:, ...]

        print "at timestep ", t, region_centers.shape[0], "traxels found"
        count = 0
        filtered_labels[t] = []
        for idx in range(region_centers.shape[0]):
            if len(region_centers[idx]) == 2:
                x, y = region_centers[idx]
                z = 0
            elif len(region_centers[idx]) == 3:
                x, y, z = region_centers[idx]
            else:
                raise Exception, "The RegionCenter feature must have dimensionality 2 or 3."
            size = pixel_count[idx]
            if (x < x_range[0] or x >= x_range[1] or
                        y < y_range[0] or y >= y_range[1] or
                        z < z_range[0] or z >= z_range[1] or
                        size < size_range[0] or size >= size_range[1]):
                filtered_labels[t].append(int(idx + 1))
                continue
            else:
                count += 1
            traxel = track.Traxel()
            traxel.set_feature_store(fs)
            traxel.set_x_scale(x_scale)
            traxel.set_y_scale(y_scale)
            traxel.set_z_scale(z_scale)
            traxel.Id = int(idx + 1)
            traxel.Timestep = int(t)

            traxel.add_feature_array("com", 3)
            for i, v in enumerate([x, y, z]):
                traxel.set_feature_value('com', i, float(v))

            if with_optical_correction:
                traxel.add_feature_array("com_corrected", 3)
                for i, v in enumerate(region_centers_corr[idx]):
                    traxel.set_feature_value("com_corrected", i, float(v))
                if len(region_centers_corr[idx]) == 2:
                    traxel.set_feature_value("com_corrected", 2, 0.)

            if with_div:
                traxel.add_feature_array("divProb", 1)
                prob = 0.0

                prob = float(divProbs[str(t)][idx + 1][1])
                # idx+1 because region_centers and pixel_count start from 1, divProbs starts from 0
                traxel.set_feature_value("divProb", 0, prob)
                division_probabilities.append(prob)

            if with_local_centers:
                raise Exception, "not yet implemented"
                traxel.add_feature_array("localCentersX", len(localCenters[t][idx + 1]))
                traxel.add_feature_array("localCentersY", len(localCenters[t][idx + 1]))
                traxel.add_feature_array("localCentersZ", len(localCenters[t][idx + 1]))
                for i, v in enumerate(localCenters[t][idx + 1]):
                    traxel.set_feature_value("localCentersX", i, float(v[0]))
                    traxel.set_feature_value("localCentersY", i, float(v[1]))
                    traxel.set_feature_value("localCentersZ", i, float(v[2]))

            if with_merger_prior and ext_probs is None:
                traxel.add_feature_array("detProb", max_num_mergers + 1)
                probs = []
                for i in range(len(detProbs[str(t)][idx + 1])):
                    probs.append(float(detProbs[str(t)][idx + 1][i]))
                probs[max_num_mergers] = sum(probs[max_num_mergers:])
                for i in range(max_num_mergers + 1):
                    traxel.set_feature_value("detProb", i, float(probs[i]))

            detection_probabilities.append([traxel.get_feature_value("detProb", i) for i in range(max_num_mergers+1)])

            traxel.add_feature_array("count", 1)
            traxel.set_feature_value("count", 0, float(size))
            if median_object_size is not None:
                obj_sizes.append(float(size))
            ts.add(fs, traxel)

        print "at timestep ", t, count, "traxels passed filter"
        max_traxel_id_at.append(int(region_centers.shape[0]))
        if count == 0:
            empty_frame = True

        total_count += count

    # load features from raw data
    if len(options.raw_filename) > 0:
        print("Computing Features from Raw Data: {}".format(options.raw_filename))
        start_time = time.time()

        with h5py.File(options.raw_filename, 'r') as raw_h5:
            shape = h5file['/'.join(options.label_img_path.split('/')[:-1])].values()[0].shape[1:4]
            shape = (len(h5file['/'.join(options.label_img_path.split('/')[:-1])].values()),) + shape
            print("Shape is {}".format(shape))

            # loop over all frames and compute features for all traxels per frame
            for timestep in xrange(max(0, time_range[0]), min(shape[0], time_range[1])):
                print("\tFrame {}".format(timestep))
                # TODO: handle smaller FOV instead of looking at full frame
                label_image_path = options.label_img_path % (timestep, timestep+1, shape[1], shape[2], shape[3])
                label_image = np.array(h5file[label_image_path][0, ..., 0]).squeeze().astype(np.uint32)
                raw_image = np.array(raw_h5['/'.join(options.raw_path.split('/'))][timestep, ..., 0]).squeeze().astype(np.float32)
                max_traxel_id = track.extract_region_features(raw_image, label_image, fs, timestep)

                # uncomment the following if no features are taken from the ilp file any more
                #
                #max_traxel_id_at.append(max_traxel_id)
                # for idx in xrange(1, max_traxel_id):
                #     traxel = track.Traxel()
                #     traxel.set_x_scale(x_scale)
                #     traxel.set_y_scale(y_scale)
                #     traxel.set_z_scale(z_scale)
                #     traxel.Id = idx
                #     traxel.Timestep = timestep
                #     ts.add(fs, traxel)

        end_time = time.time()
        print("Feature computation for a dataset of shape {} took {} secs".format(shape, end_time - start_time))
        #fs.dump()

    if median_object_size is not None:
        median_object_size[0] = np.median(np.array(obj_sizes), overwrite_input=True)
        print 'median object size = ' + str(median_object_size[0])

    return ts, fs, max_traxel_id_at, division_probabilities, detection_probabilities


def getH5Dataset(h5group, ds_name):
    if ds_name in h5group.keys():
        return np.array(h5group[ds_name])

    return np.array([])


def getRegionFeatures(ndim):
    region_features = [
            ("RegionCenter", ndim),
            ("Count", 1),
            ("Variance", 1),
            ("Sum", 1),
            ("Mean", 1),
            ("RegionRadii", ndim),
            ("Central< PowerSum<2> >", 1),
            ("Central< PowerSum<3> >", 1),
            ("Central< PowerSum<4> >", 1),
            ("Kurtosis", 1),
            ("Maximum", 1),
            ("Minimum", 1),
            ("RegionAxes", ndim**2),
            ("Skewness", 1),
            ("Weighted<PowerSum<0> >", 1),
            ("Coord< Minimum >", ndim),
            ("Coord< Maximum >", ndim)
    ]
    return region_features


def getTraxelStore(options, ilp_fn,time_range, shape):
    max_traxel_id_at = []
    with h5py.File(ilp_fn, 'r') as h5file:
        ndim = 3

        print '/'.join(options.label_img_path.strip('/').split('/')[:-1])

        if h5file['/'.join(options.label_img_path.strip('/').split('/')[:-1])].values()[0].shape[3] == 1:
            ndim = 2
        print 'ndim=', ndim

        print time_range
        if options.load_traxelstore:
            print 'loading traxelstore from file'
            import pickle

            with open(options.load_traxelstore, 'rb') as ts_in:
                ts = pickle.load(ts_in)
                fs = pickle.load(ts_in)
                max_traxel_id_at = pickle.load(ts_in)
                ts.set_feature_store(fs)

            info = [int(x) for x in ts.bounding_box()]
            t0, t1 = (info[0], info[4])
            if info[0] != options.mints or (options.maxts != -1 and info[4] != options.maxts - 1):
                if options.maxts == -1:
                    options.maxts = info[4] + 1
                print("Warning: Traxelstore has different time range than requested FOV. Trimming traxels...")
                fov = getFovFromOptions(options, shape, t0, t1)
                fov.set_time_bounds(options.mints, options.maxts - 1)
                new_ts = track.TraxelStore()
                ts.filter_by_fov(new_ts, fov)
                ts = new_ts
        else:
            max_num_mer = int(options.max_num_objects)
            ts, fs, max_traxel_id_at, division_probabilities, detection_probabilities = generate_traxelstore(h5file=h5file,
                                                            options=options,
                                                            feature_path=feature_path,
                                                            time_range=time_range,
                                                            x_range=None,
                                                            y_range=None,
                                                            z_range=None,
                                                            size_range=[options.minsize, 10000],
                                                            x_scale=options.x_scale,
                                                            y_scale=options.y_scale,
                                                            z_scale=options.z_scale,
                                                            with_div=with_div,
                                                            with_local_centers=False,
                                                            median_object_size=obj_size,
                                                            max_traxel_id_at=max_traxel_id_at,
                                                            with_merger_prior=with_merger_prior,
                                                            max_num_mergers=max_num_mer,
                                                            with_optical_correction=bool(options.woptical),
                                                            ext_probs=options.ext_probs
            )

        info = [int(x) for x in ts.bounding_box()]
        t0, t1 = (info[0], info[4])
        print "-> Traxelstore bounding box: " + str(info)

        if options.dump_traxelstore:
            print 'dumping traxelstore to file'
            import pickle

            with open(options.dump_traxelstore, 'wb') as ts_out:
                pickle.dump(ts, ts_out)
                pickle.dump(fs, ts_out)
                pickle.dump(max_traxel_id_at, ts_out)

    return ts, fs, max_traxel_id_at, ndim, t0, t1


def getFovFromOptions(options, shape, t0, t1):
    [xshape, yshape, zshape] = shape

    fov = track.FieldOfView(t0, 0, 0, 0, t1, options.x_scale * (xshape - 1), options.y_scale * (yshape - 1),
                            options.z_scale * (zshape - 1))
    return fov

def initializeConservationTracking(options, shape, t0, t1):
    ndim = 2 if shape[-1]==1 else 3
    rf_fn = 'none'
    if options.rf_fn:
        rf_fn = options.rf_fn

    fov = getFovFromOptions(options, shape, t0, t1)
    if ndim == 2:
        [xshape, yshape, zshape] = shape
        assert options.z_scale * (zshape - 1) == 0, "fov of z must be (0,0) if ndim == 2"

    if options.method == 'conservation':
        tracker = track.ConsTracking(int(options.max_num_objects),
                                     bool(options.size_dependent_detection_prob),
                                     options.avg_obj_size[0],
                                     options.mnd,
                                     not bool(options.without_divisions),
                                     options.division_threshold,
                                     rf_fn,
                                     fov,
                                     "none",
                                     track.ConsTrackingSolverType.CplexSolver)
    elif options.method == 'conservation-dynprog':
        tracker = track.ConsTracking(int(options.max_num_objects),
                                     bool(options.size_dependent_detection_prob),
                                     options.avg_obj_size[0],
                                     options.mnd,
                                     not bool(options.without_divisions),
                                     options.division_threshold,
                                     rf_fn,
                                     fov,
                                     "none",
                                     track.ConsTrackingSolverType.DynProgSolver)
    else:
        raise InvalidArgumentException("Must be conservation or conservation-dynprog")
    return tracker, fov


def negLog(features):
    fa = np.array(features)
    fa[fa < 0.0000000001] = 0.0000000001
    return list(np.log(fa) * -1.0)

def getDetectionFeatures(traxel, max_state):
    return get_feature_vector(traxel, "detProb", max_state)

def getDivisionFeatures(traxel):
    prob = traxel.get_feature_value("divProb", 0)
    return [1.0-prob, prob]

def getTransitionFeatures(traxelA, traxelB, transitionParam, max_state):
    positions = [np.array([t.X(), t.Y(), t.Z()]) for t in [traxelA, traxelB]]
    dist = np.linalg.norm(positions[0] - positions[1])
    prob = np.exp(-dist / transitionParam)
    return [1.0-prob] + [prob]*(max_state-1)

def getBoundaryCostMultiplier(traxel, fov, margin):
    dist = fov.spatial_distance_to_border(traxel.Timestep,traxel.X(),traxel.Y(),traxel.Z(), False)
    if dist > margin:
        return 1.0
    else:
        return float(dist) / margin

def listify(l):
    return [[e] for e in l]

if __name__ == "__main__":
    options, args = getConfigAndCommandLineArguments()

    # get filenames
    numArgs = len(args)
    fns = []
    if numArgs > 0:
        for arg in args:
            print arg
            fns.extend(glob.glob(arg))
        fns.sort()
        print(fns)

    print fns
    ilp_fn = fns[0]

    # create output path
    if not path.exists(options.out_dir):
        try:
            os.makedirs(options.out_dir)
        except:
            pass

    ### Do the tracking
    start = time.time()

    feature_path = options.feats_path
    with_div = not bool(options.without_divisions)
    with_merger_prior = True

    # get selected time range
    time_range = [options.mints, options.maxts]
    if options.maxts == -1 and options.mints == 0:
        time_range = None

    # set average object size if chosen
    obj_size = [0]
    if options.avg_obj_size != 0:
        obj_size[0] = options.avg_obj_size
    else:
        options.avg_obj_size = obj_size

    # find shape of dataset
    with h5py.File(ilp_fn, 'r') as h5file:
        shape = h5file['/'.join(options.label_img_path.split('/')[:-1])].values()[0].shape[1:4]

    ts, fs, max_traxel_id_at, ndim, t0, t1 = getTraxelStore(options, ilp_fn, time_range, shape)

    # initialize tracker to get hypotheses graph
    tracker, fov = initializeConservationTracking(options, shape, t0, t1)
    hypotheses_graph = tracker.buildGraph(ts, options.max_nearest_neighbors)

    # create tracklet graph if necessary
    if not options.without_tracklets:
        traxel_graph = hypotheses_graph
        hypotheses_graph = traxel_graph.generate_tracklet_graph()

    n_it = track.NodeIt(hypotheses_graph)
    a_it = track.ArcIt(hypotheses_graph)
    
    # get the map of node -> list(traxel) or just traxel
    if options.without_tracklets:
        traxelMap = hypotheses_graph.getNodeTraxelMap()
    else:
        traxelMap = hypotheses_graph.getNodeTrackletMap()

    jsonRoot = {}
    traxelIdPerTimestepToUniqueIdMap = {}

    numElements = track.countNodes(hypotheses_graph) + track.countArcs(hypotheses_graph)
    progressBar = ProgressBar(stop=numElements)
    maxNumObjects = int(options.max_num_objects) + 1 # because we need features for state 0 as well!
    margin = float(options.border_width)

    # add all detections
    detections = []
    for i,n in enumerate(n_it):
        detection = {}
        if options.without_tracklets:
            # only one traxel, but make it a list so everything below works the same
            traxels = [traxelMap[n]]
        else:
            traxels = traxelMap[n]

        # store mapping of all contained traxels to this detection uuid
        for t in traxels:
            traxelIdPerTimestepToUniqueIdMap.setdefault(str(t.Timestep), {})[str(t.Id)] = i

        detection['id'] = i

        # accumulate features over all contained traxels
        previousTraxel = None
        detFeats = np.zeros(maxNumObjects)
        for t in traxels:
            detFeats += np.array(negLog(getDetectionFeatures(t, maxNumObjects)))
            if previousTraxel is not None:
                detFeats += np.array(negLog(getTransitionFeatures(previousTraxel, t, options.trans_par, maxNumObjects)))
            previousTraxel = t

        detection['features'] = listify(list(detFeats))

        # division only if probability is big enough
        try:
            divFeats = getDivisionFeatures(traxels[-1])
            if divFeats[1] > options.division_threshold:
                detection['divisionFeatures'] = listify(negLog(divFeats))
        except:
            pass
        
        if traxels[0].Timestep <= t0:
            detection['appearanceFeatures'] = listify([0.0]*(maxNumObjects))
        else:
            detection['appearanceFeatures'] = listify([0.0] + [float(options.appearance_cost) * getBoundaryCostMultiplier(traxels[0], fov, margin)] * (maxNumObjects-1))
        
        if traxels[-1].Timestep >= t1:
            detection['disappearanceFeatures'] = listify([0.0]*(maxNumObjects))
        else:
            detection['disappearanceFeatures'] = listify([0.0] + [float(options.disappearance_cost) * getBoundaryCostMultiplier(traxels[-1], fov, margin)] * (maxNumObjects-1))

        detections.append(detection)
        progressBar.show()

    # add all links
    links = []
    for a in a_it:
        link = {}
        if options.without_tracklets:
            srcTraxel = traxelMap[hypotheses_graph.source(a)]
            destTraxel = traxelMap[hypotheses_graph.target(a)]
        else:
            srcTraxel = traxelMap[hypotheses_graph.source(a)][-1] # src is last of the traxels in source tracklet
            destTraxel = traxelMap[hypotheses_graph.target(a)][0] # dest is first of traxels in destination tracklet
        link['src'] = traxelIdPerTimestepToUniqueIdMap[str(srcTraxel.Timestep)][str(srcTraxel.Id)]
        link['dest'] = traxelIdPerTimestepToUniqueIdMap[str(destTraxel.Timestep)][str(destTraxel.Id)]
        link['features'] = listify(negLog(getTransitionFeatures(srcTraxel, destTraxel, options.trans_par, maxNumObjects)))
        links.append(link)
        progressBar.show()

    settings = {}
    settings['statesShareWeights'] = True
    settings['allowPartialMergerAppearance'] = False
    settings['requireSeparateChildrenOfDivision'] = True
    settings['optimizerEpGap'] = 0.01
    settings['optimizerVerbose'] = False
    settings['optimizerNumThreads'] = 1

    jsonRoot['traxelToUniqueId'] = traxelIdPerTimestepToUniqueIdMap
    jsonRoot['segmentationHypotheses'] = detections
    jsonRoot['linkingHypotheses'] = links
    jsonRoot['settings'] = settings
    jsonRoot['exclusions'] = []

    with open(options.json_filename, 'w') as f:
        json.dump(jsonRoot, f, indent=4, separators=(',', ': '))

