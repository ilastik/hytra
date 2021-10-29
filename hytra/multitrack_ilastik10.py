import sys
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
import copy
import pgmlink as track

sys.path.append(path.join(path.dirname(__file__), path.pardir))
sys.path.append(path.dirname(__file__))

from hytra.core.trackingfeatures import (
    extract_features_and_compute_score,
    get_feature_vector,
)
from hytra.util.progressbar import ProgressBar
from empryonic import io
from multiprocessing import Pool


def swirl_motion_func_creator(velocityWeight):
    def swirl_motion_func(traxelA, traxelB, traxelC, traxelD):
        # print("SwirlMotion evaluated for traxels: {}, {}, {}".format(traxelA, traxelB, traxelC))
        traxels = [traxelA, traxelB, traxelC, traxelD]
        positions = [np.array([t.X(), t.Y(), t.Z()]) for t in traxels]
        vecs = [positions[1] - positions[0], positions[2] - positions[1]]

        # acceleration is change in velocity
        acc = vecs[1] - vecs[0]

        # assume constant acceleration to find expected velocity vector
        expected_vel = vecs[1] + acc

        # construct expected position
        expected_pos = positions[2] + expected_vel

        # penalize deviation from that position
        deviation = np.linalg.norm(expected_pos - positions[3])
        # print("\tExpected next traxel at pos {}, but found {}. Distance={}".format(expected_pos, positions[3], deviation))
        cost = float(velocityWeight) * deviation
        tIds = [(t.Timestep, t.Id) for t in traxels]
        # print("Adding cost {} to link between traxels {} at positions {}".format(cost, tIds, positions))

        return cost

    return swirl_motion_func


def getConfigAndCommandLineArguments():

    usage = """%prog [options] FILES
Track cells.

Before processing, input files are copied to OUT_DIR. Groups, that will not be modified are not
copied but linked to the original files to improve execution speed and storage requirements.
"""

    parser = optparse.OptionParser(usage=usage)
    parser.add_option(
        "--config-file",
        type="string",
        dest="config",
        default=None,
        help="path to config file",
    )
    parser.add_option(
        "--method",
        type="str",
        default="conservation",
        help="conservation, conservation-twostage or conservation-dynprog [default: %default]",
    )
    parser.add_option(
        "-o",
        "--output-dir",
        type="str",
        dest="out_dir",
        default="tracked",
        help="[default: %default]",
    )
    parser.add_option(
        "--x-scale",
        type="float",
        dest="x_scale",
        default=1.0,
        help="[default: %default]",
    )
    parser.add_option(
        "--y-scale",
        type="float",
        dest="y_scale",
        default=1.0,
        help="[default: %default]",
    )
    parser.add_option(
        "--z-scale",
        type="float",
        dest="z_scale",
        default=1.0,
        help="[default: %default]",
    )
    parser.add_option(
        "--with-visbricks",
        action="store_true",
        dest="with_visbricks",
        help='writze out a "time dependent HDF5" file for visbricks',
    )
    parser.add_option(
        "--with-rel-linking",
        action="store_true",
        dest="with_rel_linking",
        help="link hdf5 files relative instead of absolute",
    )
    parser.add_option(
        "--full-copy",
        action="store_true",
        dest="full_copy",
        help="do not link to but copy input files completely",
    )
    parser.add_option(
        "--user",
        type="str",
        dest="user",
        default=getpass.getuser(),
        help="user to log [default: %default]",
    )
    parser.add_option(
        "--date",
        type="str",
        dest="date",
        default=time.ctime(),
        help="datetime to log [default: %default]",
    )
    parser.add_option(
        "--machine",
        type="str",
        dest="machine",
        default=socket.gethostname(),
        help="machine to log [default: %default]",
    )
    parser.add_option(
        "--comment",
        type="str",
        dest="comment",
        default="none",
        help="some comment to log [default: %default]",
    )
    parser.add_option(
        "--random-forest",
        type="string",
        dest="rf_fn",
        default=None,
        help="use cellness prediction instead of indicator function for (mis-)detection energy",
    )
    parser.add_option(
        "--ep_gap",
        type="float",
        dest="ep_gap",
        default=0.01,
        help="stop optimization as soon as a feasible integer solution is found proved to be within the given percent of the optimal solution",
    )
    parser.add_option(
        "--num-threads",
        type=int,
        dest="num_threads",
        default=0,
        help="Number of threads for CPLEX, 0 -> number of cores",
    )
    parser.add_option(
        "-f",
        "--forbidden_cost",
        type="float",
        dest="forbidden_cost",
        default=0,
        help="forbidden cost [default: %default]",
    )
    parser.add_option(
        "--min-ts", type="int", dest="mints", default=0, help="[default: %default]"
    )
    parser.add_option(
        "--max-ts", type="int", dest="maxts", default=-1, help="[default: %default]"
    )
    parser.add_option(
        "--min-size",
        type="int",
        dest="minsize",
        default=0,
        help="minimal size of objects to be tracked [default: %default]",
    )
    parser.add_option(
        "--dump-traxelstore",
        type="string",
        dest="dump_traxelstore",
        default=None,
        help="dump traxelstore to file [default: %default]",
    )
    parser.add_option(
        "--load-traxelstore",
        type="string",
        dest="load_traxelstore",
        default=None,
        help="load traxelstore from file [default: %default]",
    )
    parser.add_option(
        "--raw-data-file",
        type="string",
        dest="raw_filename",
        default="",
        help="filename to the raw h5 file",
    )
    parser.add_option(
        "--raw-data-path",
        type="string",
        dest="raw_path",
        default="volume/data",
        help="Path inside the raw h5 file to the data",
    )
    parser.add_option(
        "--dump-hypotheses-graph",
        type="string",
        dest="hypotheses_graph_filename",
        default=None,
        help="save hypotheses graph so it can be loaded later",
    )

    # funkey learning parameters
    parser.add_option(
        "--export-funkey-files",
        action="store_true",
        dest="export_funkey",
        default=False,
        help="export labels features and constraints to learn with funkey [default=%default]",
    )
    parser.add_option(
        "--learn-perturbation-weights",
        action="store_true",
        dest="learn_perturbation_weights",
        default=False,
        help="learn free parameters of perturbation models [default=%default]",
    )
    parser.add_option(
        "--without-classifier",
        action="store_true",
        dest="woc",
        default=False,
        help="use classifier for labeling [default=%default]",
    )
    parser.add_option(
        "--gt-path",
        type="string",
        dest="gt_pth",
        default="",
        help="path to ground truth files [default=%default]",
    )
    parser.add_option(
        "--gt-path-format",
        type="string",
        dest="gt_path_format_string",
        default="{0:04d}",
        help="groundtruth file format string, should be {0:05d} or {0:04d} [default=%default]",
    )
    parser.add_option(
        "--funkey-learn",
        action="store_true",
        dest="learn_funkey",
        default=False,
        help="export labels features and constraints to learn with funkey [default=%default]",
    )
    parser.add_option(
        "--funkey-weights",
        type="string",
        dest="funkey_weights",
        default="",
        help="weights for weighted hamming loss [default=%default]",
    )
    parser.add_option(
        "--funkey-regularizerWeight",
        type="float",
        dest="funkey_regularizerWeight",
        default=1,
        help="stop optimization as soon as a feasible integer solution is found \
                     proved to be within the given percent of the optimal solution [default=%default]",
    )
    parser.add_option("--compare-script-path", type="string", dest="compare_path")
    parser.add_option(
        "--only-labels",
        action="store_true",
        dest="only_labels",
        help="skip exporting feature and constraint files",
    )

    consopts = optparse.OptionGroup(parser, "conservation tracking")
    consopts.add_option(
        "--max-number-objects",
        dest="max_num_objects",
        type="float",
        default=2,
        help="Give maximum number of objects one connected component may consist of [default: %default]",
    )
    consopts.add_option(
        "--max-neighbor-distance",
        dest="mnd",
        type="float",
        default=30,
        help="[default: %default]",
    )
    consopts.add_option(
        "--max-nearest-neighbors",
        dest="max_nearest_neighbors",
        type="int",
        default=1,
        help="[default: %default]",
    )
    consopts.add_option(
        "--division-threshold",
        dest="division_threshold",
        type="float",
        default=0.1,
        help="[default: %default]",
    )
    # detection_rf_filename in general parser options
    consopts.add_option(
        "--size-dependent-detection-prob",
        dest="size_dependent_detection_prob",
        action="store_true",
    )
    # forbidden_cost in general parser options
    # ep_gap in general parser options
    consopts.add_option(
        "--average-obj-size",
        dest="avg_obj_size",
        type="float",
        default=0,
        help="[default: %default]",
    )
    consopts.add_option(
        "--without-tracklets", dest="without_tracklets", action="store_true"
    )
    consopts.add_option("--with-opt-correct", dest="woptical", action="store_true")
    consopts.add_option(
        "--det",
        dest="detection_weight",
        type="float",
        default=10.0,
        help="detection weight [default: %default]",
    )
    consopts.add_option(
        "--div",
        dest="division_weight",
        type="float",
        default=10.0,
        help="division weight [default: %default]",
    )
    consopts.add_option(
        "--dis",
        dest="disappearance_cost",
        type="float",
        default=500.0,
        help="disappearance cost [default: %default]",
    )
    consopts.add_option(
        "--app",
        dest="appearance_cost",
        type="float",
        default=500.0,
        help="appearance cost [default: %default]",
    )
    consopts.add_option(
        "--tr",
        dest="transition_weight",
        type="float",
        default=10.0,
        help="transition weight [default: %default]",
    )
    consopts.add_option(
        "--motionModelWeight",
        dest="motionModelWeight",
        type="float",
        default=0.0,
        help="motion model weight [default: %default]",
    )
    consopts.add_option(
        "--without-divisions", dest="without_divisions", action="store_true"
    )
    consopts.add_option(
        "--means",
        dest="means",
        type="float",
        default=0.0,
        help="means for detection [default: %default]",
    )
    consopts.add_option(
        "--sigma",
        dest="sigma",
        type="float",
        default=0.0,
        help="sigma for detection [default: %default]",
    )
    consopts.add_option(
        "--with-merger-resolution",
        dest="with_merger_resolution",
        action="store_true",
        default=False,
    )
    consopts.add_option(
        "--without-constraints", dest="woconstr", action="store_true", default=False
    )
    consopts.add_option(
        "--trans-par",
        dest="trans_par",
        type="float",
        default=5.0,
        help="alpha for the transition prior [default: %default]",
    )
    consopts.add_option(
        "--border-width",
        dest="border_width",
        type="float",
        default=0.0,
        help="absolute border margin in which the appearance/disappearance costs are linearly decreased [default: %default]",
    )
    consopts.add_option(
        "--ext-probs",
        dest="ext_probs",
        type="string",
        default=None,
        help="provide a path to hdf5 files containing detection probabilities [default:%default]",
    )
    consopts.add_option(
        "--objCountPath",
        dest="obj_count_path",
        type="string",
        default="/CellClassification/Probabilities/0/",
        help="internal hdf5 path to object count probabilities [default=%default]",
    )
    consopts.add_option(
        "--divPath",
        dest="div_prob_path",
        type="string",
        default="/DivisionDetection/Probabilities/0/",
        help="internal hdf5 path to division probabilities [default=%default]",
    )
    consopts.add_option(
        "--featsPath",
        dest="feats_path",
        type="string",
        default="/ObjectExtraction/RegionFeatures/0/[[%d], [%d]]/Default features/%s",
        help="internal hdf5 path to object features [default=%default]",
    )
    consopts.add_option(
        "--translationPath",
        dest="trans_vector_path",
        type="str",
        default="OpticalTranslation/TranslationVectors/0/data",
        help="internal hdf5 path to translation vectors [default=%default]",
    )
    consopts.add_option(
        "--labelImgPath",
        dest="label_img_path",
        type="str",
        default="/ObjectExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]",
        help="internal hdf5 path to label image [default=%default]",
    )
    consopts.add_option(
        "--timeout",
        dest="timeout",
        type="float",
        default=1e75,
        help="CPLEX timeout in sec. [default: %default]",
    )
    consopts.add_option(
        "--with-graph-labeling",
        dest="w_labeling",
        action="store_true",
        default=False,
        help="load ground truth labeling into hypotheses graph for further evaluation on C++ side,\
                        requires gt-path to point to the groundtruth files",
    )
    consopts.add_option(
        "--without-swaps", dest="without_swaps", action="store_true", default=False
    )
    consopts.add_option(
        "--max-num-paths",
        dest="max_num_paths",
        type="int",
        default=None,
        help="Max number of paths DynProg is allowed to find",
    )

    # options for perturbing and re-ranking
    consopts.add_option(
        "--num-iterations",
        dest="num_iterations",
        type="int",
        default=1,
        help="number of iterations to perform inference, >1 needed for uncertainty [default=%default]",
    )
    consopts.add_option(
        "--perturb-distrib",
        dest="perturbation_distribution",
        type="str",
        default="DiverseMbest",
        help="distribution type for parameter perturbation, one of {GaussianPertubation, PerturbAndMAP, "
        "DiverseMbest, MbestCPLEX, ClassifierUncertainty} [default=%default]",
    )
    consopts.add_option(
        "--perturb-sigmas",
        dest="perturb_sigmas",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 10.0, 10.0, 10.0],
        help="Parameters for the perturbation distribution. [default=%default]",
    )
    consopts.add_option(
        "--objCountVarPath",
        dest="obj_count_var_path",
        type="string",
        default="/CountClassification/Uncertainty/0/",
        help="internal path to count classification variance [default=%default]",
    )
    consopts.add_option(
        "--divVarPath",
        dest="div_prob_var_path",
        type="string",
        default="/DivisionDetection/Uncertainty/0/",
        help="internal path to division detection variance [default=%default]",
    )
    consopts.add_option(
        "--transitionClassifierFn",
        dest="trans_fn",
        type="string",
        default=None,
        help="filename of transition classifier, empty string for None [default=%default]",
    )
    consopts.add_option(
        "--transitionClassifierPath",
        dest="trans_path",
        type="string",
        default="/TransitionClassifier",
        help="internal h5 path to transition classifier [default=%default]",
    )

    # learning options
    consopts.add_option(
        "--no-saving",
        dest="skip_saving",
        action="store_true",
        help="Skip saving if you are just interested in the proposals and features",
    )
    consopts.add_option(
        "--only-save-first",
        dest="save_first",
        action="store_true",
        help="Skip saving of all perturbations, except the unperturbed result",
    )
    consopts.add_option(
        "--save-outlier-svm",
        dest="save_outlier_svm",
        type="str",
        default="",
        help="Filename to store the trained SVM for higher order features outlier detection. Requires gt-path to be set",
    )
    consopts.add_option(
        "--load-outlier-svm",
        dest="load_outlier_svm",
        type="str",
        default="",
        help="Load a trained outlier svm from disk",
    )
    consopts.add_option(
        "--reranker-weight-file",
        dest="reranker_weight_file",
        type="str",
        default="",
        help="File in which the reranker stored the weights after learning",
    )

    parser.add_option_group(consopts)

    optcfg, args = parser.parse_args()

    if optcfg.config != None:
        with open(optcfg.config) as f:
            configfilecommands = f.read().splitlines()
        optcfg, args2 = parser.parse_args(configfilecommands)

    print("--------------")
    for key, value in sorted(vars(optcfg).items()):
        print("{} : {}".format(key, value))
    print("--------------")

    numArgs = len(args)
    if numArgs == 0:
        parser.print_help()
        sys.exit(1)

    if optcfg.learn_funkey:
        assert optcfg.export_funkey, "cannot learn funkey without exporting the files"

    return optcfg, args


def extract_coordinates(coordinate_map, h5file, traxel, options):
    # add coordinate lists with armadillo matrixes
    shape = list(h5file["/".join(options.label_img_path.split("/")[:-1])].values())[
        0
    ].shape[1:4]
    ndim = 2 if shape[-1] == 1 else 3

    # print("Extracting coordinates of potential merger: timestep {} id {}".format(traxel.Timestep, traxel.Id))

    lower = get_feature_vector(traxel, "Coord< Minimum >", ndim)
    upper = get_feature_vector(traxel, "Coord< Maximum >", ndim)

    limg_path_at = options.label_img_path % tuple(
        [traxel.Timestep, traxel.Timestep + 1] + list(shape)
    )
    roi = [0] * 3
    roi[0] = slice(int(lower[0]), int(upper[0] + 1))
    roi[1] = slice(int(lower[1]), int(upper[1] + 1))
    if ndim == 3:
        roi[2] = slice(int(lower[2]), int(upper[2] + 1))
    else:
        assert ndim == 2
    image_excerpt = np.array(h5file[limg_path_at][tuple([0] + roi + [0])])
    try:
        track.extract_coordinates(
            coordinate_map,
            image_excerpt.astype(np.uint32),
            np.array(lower).astype(np.int64),
            traxel,
        )
    except:
        print(
            "Could not run extract_coordinates for traxel: Id={} Timestep={}".format(
                traxel.Id, traxel.Timestep
            )
        )
        raise Exception


def update_merger_features(
    coordinate_map,
    h5file,
    merger_traxel,
    new_traxel_ids,
    raw_h5,
    options,
    fs,
    ts,
    timestep,
):
    # add coordinate lists with armadillo matrixes
    shape = list(h5file["/".join(options.label_img_path.split("/")[:-1])].values())[
        0
    ].shape[1:4]
    ndim = 2 if shape[-1] == 1 else 3

    # print("Updating label image of merger traxel: timestep {} id {}".format(merger_traxel.Timestep, merger_traxel.Id))

    lower = get_feature_vector(merger_traxel, "Coord< Minimum >", ndim)
    upper = get_feature_vector(merger_traxel, "Coord< Maximum >", ndim)

    limg_path_at = options.label_img_path % tuple(
        [merger_traxel.Timestep, merger_traxel.Timestep + 1] + list(shape)
    )
    roi = [0] * 3
    roi[0] = slice(int(lower[0]), int(upper[0] + 1))
    roi[1] = slice(int(lower[1]), int(upper[1] + 1))
    if ndim == 3:
        roi[2] = slice(int(lower[2]), int(upper[2] + 1))
    else:
        assert ndim == 2

    label_image_excerpt = np.array(h5file[limg_path_at][tuple([0] + roi + [0])])
    min_new_traxel_id = min(new_traxel_ids)
    label_image_excerpt = np.zeros(label_image_excerpt.shape, dtype=np.uint32) + (
        min_new_traxel_id - 1
    )

    try:
        for traxel_id in new_traxel_ids:
            label_image_excerpt = track.update_labelimage(
                coordinate_map,
                label_image_excerpt,
                ts,
                np.array(lower).astype(np.int64),
                int(merger_traxel.Timestep),
                int(traxel_id),
            )
            if traxel_id not in label_image_excerpt:
                print(
                    "ERROR: could not find requested traxel_id {} in relabeled image {}".format(
                        traxel_id, label_image_excerpt
                    )
                )
        label_image_excerpt -= min_new_traxel_id - 1
    except MemoryError:
        print(
            "WARNING: Something went wrong when updating label image around traxel "
            "(id={}, timestep={}).".format(merger_traxel.Id, merger_traxel.Timestep)
        )

    try:
        # compute features for new traxels, which are automatically stored in featurestore
        # the 2d raw sequence is usually saved as a 2d+t+c dataset, the labelimage from ilastik as 3d+t+c
        raw_image_excerpt = np.array(
            raw_h5[options.raw_path][tuple([timestep] + roi[0:ndim] + [0])]
        )
        ri_ex = raw_image_excerpt.astype(np.float32)
        if ndim == 2:
            ri_ex = ri_ex.squeeze()
        track.extract_region_features_roi(
            ri_ex,
            label_image_excerpt.astype(np.uint32),
            np.array(range(len(new_traxel_ids) + 1)).astype(np.int64),
            min_new_traxel_id - 1,
            np.array(lower).astype(np.int64),
            fs,
            merger_traxel.Timestep,
        )
        # make sure that the traxel is not too small, otherwise copy skewness, kurtosis and variance from parent
        def copy_feature_from_parent(traxel, feature_name):
            val = get_feature_vector(merger_traxel, feature_name, 1)
            traxel.set_feature_value(feature_name, 0, val[0])

        for traxel_id in new_traxel_ids:
            traxel = ts.get_traxel(int(traxel_id), int(merger_traxel.Timestep))
            t_count = get_feature_vector(traxel, "Count", 1)[0]
            if t_count < 4:
                copy_feature_from_parent(traxel, "Kurtosis")
            if t_count < 3:
                copy_feature_from_parent(traxel, "Skewness")
            if t_count < 2:
                copy_feature_from_parent(traxel, "Variance")

    except MemoryError:
        print(
            "WARNING: Something went wrong when extracting region features for traxel "
            "(id={}, timestep={}) in region {} of image size {}. Features are left empty".format(
                merger_traxel.Id, merger_traxel.Timestep, roi, ri_ex.shape
            )
        )


def generate_traxelstore(
    h5file,
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
    ext_probs=None,
):
    print("generating traxels")
    print("filling traxelstore")
    ts = track.TraxelStore()
    fs = track.FeatureStore()
    max_traxel_id_at = track.VectorOfInt()

    print("fetching region features and division probabilities")
    print(h5file.filename, feature_path)

    if with_div:
        print(options.div_prob_path)
        divProbs = h5file[options.div_prob_path]

    if with_merger_prior:
        detProbs = h5file[options.obj_count_path]

    if with_local_centers:
        localCenters = None  # self.RegionLocalCenters(time_range).wait()

    if options.perturbation_distribution == "ClassifierUncertainty":
        detProb_Vars = h5file[options.obj_count_var_path]
        if with_div:
            divProb_Vars = h5file[options.div_prob_var_path]

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
        keys_sorted = [
            key for key in keys_sorted if time_range[0] <= int(key) < time_range[1]
        ]
    else:
        time_range = (0, shape_t)

    filtered_labels = {}
    obj_sizes = []
    total_count = 0
    empty_frame = False

    for t in keys_sorted:
        feats_name = options.feats_path % (t, t + 1, "RegionCenter")
        # region_centers = np.array(feats[t]['0']['RegionCenter'])
        region_centers = np.array(h5file[feats_name])

        feats_name = options.feats_path % (t, t + 1, "Coord<Minimum>")
        lower = np.array(h5file[feats_name])
        feats_name = options.feats_path % (t, t + 1, "Coord<Maximum>")
        upper = np.array(h5file[feats_name])

        if region_centers.size:
            region_centers = region_centers[1:, ...]
            lower = lower[1:, ...]
            upper = upper[1:, ...]
        if with_optical_correction:
            try:
                feats_name = options.feats_path % (t, t + 1, "RegionCenter_corr")
                region_centers_corr = np.array(h5file[feats_name])
            except:
                raise Exception(
                    "cannot consider optical correction since it has not been computed"
                )
            if region_centers_corr.size:
                region_centers_corr = region_centers_corr[1:, ...]

        feats_name = options.feats_path % (t, t + 1, "Count")
        # pixel_count = np.array(feats[t]['0']['Count'])
        pixel_count = np.array(h5file[feats_name])
        if pixel_count.size:
            pixel_count = pixel_count[1:, ...]

        print("at timestep ", t, region_centers.shape[0], "traxels found")
        count = 0
        filtered_labels[t] = []
        for idx in range(region_centers.shape[0]):
            if len(region_centers[idx]) == 2:
                x, y = region_centers[idx]
                z = 0
            elif len(region_centers[idx]) == 3:
                x, y, z = region_centers[idx]
            else:
                raise Exception(
                    "The RegionCenter feature must have dimensionality 2 or 3."
                )
            size = pixel_count[idx]
            if (
                x < x_range[0]
                or x >= x_range[1]
                or y < y_range[0]
                or y >= y_range[1]
                or z < z_range[0]
                or z >= z_range[1]
                or size < size_range[0]
                or size >= size_range[1]
            ):
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
                traxel.set_feature_value("com", i, float(v))

            if with_optical_correction:
                traxel.add_feature_array("com_corrected", 3)
                for i, v in enumerate(region_centers_corr[idx]):
                    traxel.set_feature_value("com_corrected", i, float(v))
                if len(region_centers_corr[idx]) == 2:
                    traxel.set_feature_value("com_corrected", 2, 0.0)

            if with_div:
                traxel.add_feature_array("divProb", 1)
                prob = 0.0

                prob = float(divProbs[str(t)][idx + 1][1])
                # idx+1 because region_centers and pixel_count start from 1, divProbs starts from 0
                traxel.set_feature_value("divProb", 0, prob)

            if with_local_centers:
                raise Exception("not yet implemented")
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

            if options.perturbation_distribution == "ClassifierUncertainty":
                traxel.add_feature_array(
                    "detProb_Var", len(detProb_Vars[str(t)][idx + 1])
                )
                for i in range(len(detProb_Vars[str(t)][idx + 1])):
                    traxel.set_feature_value(
                        "detProb_Var", i, float(detProb_Vars[str(t)][idx + 1][i])
                    )

                if with_div:
                    traxel.add_feature_array(
                        "divProb_Var", len(divProb_Vars[str(t)][idx + 1])
                    )
                    for i in range(len(divProb_Vars[str(t)][idx + 1])):
                        traxel.set_feature_value(
                            "divProb_Var", i, float(divProb_Vars[str(t)][idx + 1][i])
                        )

            elif with_merger_prior and ext_probs is not None:
                assert max_num_mergers == 1, "not implemented for max_num_mergers > 1"
                detProbFilename = ext_probs % t
                detProbGroup = h5py.File(detProbFilename, "r")["objects/meta"]
                traxel_index = np.where(detProbGroup["id"][()] == traxel.Id)[0][0]
                detProbFeat = [
                    detProbGroup["prediction_class0"][()][traxel_index],
                    detProbGroup["prediction_class1"][()][traxel_index],
                ]
                traxel.add_feature_array("detProb", 2)
                for i in xrange(len(detProbFeat)):
                    traxel.set_feature_value("detProb", i, float(detProbFeat[i]))

            traxel.add_feature_array("count", 1)
            traxel.set_feature_value("count", 0, float(size))
            if median_object_size is not None:
                obj_sizes.append(float(size))
            ts.add(fs, traxel)

        print("at timestep ", t, count, "traxels passed filter")
        max_traxel_id_at.append(int(region_centers.shape[0]))
        if count == 0:
            empty_frame = True

        total_count += count

    # load features from raw data
    if len(options.raw_filename) > 0:
        print("Computing Features from Raw Data: {}".format(options.raw_filename))
        start_time = time.time()

        with h5py.File(options.raw_filename, "r") as raw_h5:
            shape = list(
                h5file["/".join(options.label_img_path.split("/")[:-1])].values()
            )[0].shape[1:4]
            shape = (
                len(h5file["/".join(options.label_img_path.split("/")[:-1])].values()),
            ) + shape
            print("Shape is {}".format(shape))

            # loop over all frames and compute features for all traxels per frame
            for timestep in xrange(max(0, time_range[0]), min(shape[0], time_range[1])):
                print("\tFrame {}".format(timestep))
                # TODO: handle smaller FOV instead of looking at full frame
                label_image_path = options.label_img_path % (
                    timestep,
                    timestep + 1,
                    shape[1],
                    shape[2],
                    shape[3],
                )
                label_image = (
                    np.array(h5file[label_image_path][0, ..., 0])
                    .squeeze()
                    .astype(np.uint32)
                )
                raw_image = (
                    np.array(
                        raw_h5["/".join(options.raw_path.split("/"))][timestep, ..., 0]
                    )
                    .squeeze()
                    .astype(np.float32)
                )
                max_traxel_id = track.extract_region_features(
                    raw_image, label_image, fs, timestep
                )

                # uncomment the following if no features are taken from the ilp file any more
                #
                # max_traxel_id_at.append(max_traxel_id)
                # for idx in xrange(1, max_traxel_id):
                #     traxel = track.Traxel()
                #     traxel.set_x_scale(x_scale)
                #     traxel.set_y_scale(y_scale)
                #     traxel.set_z_scale(z_scale)
                #     traxel.Id = idx
                #     traxel.Timestep = timestep
                #     ts.add(fs, traxel)

        end_time = time.time()
        print(
            "Feature computation for a dataset of shape {} took {} secs".format(
                shape, end_time - start_time
            )
        )
        # fs.dump()

    if median_object_size is not None:
        median_object_size[0] = np.median(np.array(obj_sizes), overwrite_input=True)
        print("median object size = " + str(median_object_size[0]))

    return ts, fs, max_traxel_id_at  # , filtered_labels, empty_frame


def write_detections(detections, fn):
    with io.LineageH5(fn, "r") as f:
        traxel_ids = f["objects/meta/id"][()]
        valid = f["objects/meta/valid"][()]
    assert len(traxel_ids) == len(valid)
    assert len(detections) == len(np.flatnonzero(valid))

    detection_indicator = np.zeros(len(traxel_ids), dtype=np.uint16)

    for i, traxel_id in enumerate(traxel_ids):
        if valid[i] != 0:
            if detections[int(traxel_id)]:
                detection_indicator[i] = 1
            else:
                detection_indicator[i] = 0

    with io.LineageH5(fn, "r+") as f:
        # delete old dataset
        if "detection" in f["objects/meta"].keys():
            del f["objects/meta/detection"]
        f.create_dataset("objects/meta/detection", data=detection_indicator)


def write_events(events, fn):
    dis = []
    app = []
    div = []
    mov = []
    mer = []
    mul = []
    print("-- Writing results to " + fn)
    for event in events:
        if event.type == track.EventType.Appearance:
            app.append((event.traxel_ids[0], event.energy))
        if event.type == track.EventType.Disappearance:
            dis.append((event.traxel_ids[0], event.energy))
        if event.type == track.EventType.Division:
            div.append(
                (
                    event.traxel_ids[0],
                    event.traxel_ids[1],
                    event.traxel_ids[2],
                    event.energy,
                )
            )
        if event.type == track.EventType.Move:
            mov.append((event.traxel_ids[0], event.traxel_ids[1], event.energy))
        if event.type == track.EventType.Merger:
            mer.append((event.traxel_ids[0], event.traxel_ids[1], event.energy))
        if event.type == track.EventType.MultiFrameMove:
            mul.append(tuple(event.traxel_ids) + (event.energy,))

    # convert to ndarray for better indexing
    dis = np.asarray(dis)
    app = np.asarray(app)
    div = np.asarray(div)
    mov = np.asarray(mov)
    mer = np.asarray(mer)
    mul = np.asarray(mul)

    # write only if file exists
    with io.LineageH5(fn, "r+") as f_curr:
        # delete old tracking
        if "tracking" in f_curr.keys():
            del f_curr["tracking"]

        tg = f_curr.create_group("tracking")

        # write associations
        if len(app):
            ds = tg.create_dataset("Appearances", data=app[:, :-1], dtype=np.int32)
            ds.attrs["Format"] = "cell label appeared in current file"

            ds = tg.create_dataset(
                "Appearances-Energy", data=app[:, -1], dtype=np.double
            )
            ds.attrs["Format"] = "lower energy -> higher confidence"

        if len(dis):
            ds = tg.create_dataset("Disappearances", data=dis[:, :-1], dtype=np.int32)
            ds.attrs["Format"] = "cell label disappeared in current file"

            ds = tg.create_dataset(
                "Disappearances-Energy", data=dis[:, -1], dtype=np.double
            )
            ds.attrs["Format"] = "lower energy -> higher confidence"

        if len(mov):
            ds = tg.create_dataset("Moves", data=mov[:, :-1], dtype=np.int32)
            ds.attrs["Format"] = "from (previous file), to (current file)"

            ds = tg.create_dataset("Moves-Energy", data=mov[:, -1], dtype=np.double)
            ds.attrs["Format"] = "lower energy -> higher confidence"

        if len(div):
            ds = tg.create_dataset("Splits", data=div[:, :-1], dtype=np.int32)
            ds.attrs[
                "Format"
            ] = "ancestor (previous file), descendant (current file), descendant (current file)"

            ds = tg.create_dataset("Splits-Energy", data=div[:, -1], dtype=np.double)
            ds.attrs["Format"] = "lower energy -> higher confidence"

        if len(mer):
            ds = tg.create_dataset("Mergers", data=mer[:, :-1], dtype=np.int32)
            ds.attrs["Format"] = "descendant (current file), number of objects"

            ds = tg.create_dataset("Mergers-Energy", data=mer[:, -1], dtype=np.double)
            ds.attrs["Format"] = "lower energy -> higher confidence"

        if len(mul):
            ds = tg.create_dataset("MultiFrameMoves", data=mul[:, :-1], dtype=np.int32)
            ds.attrs["Format"] = "from (given by timestep), to (current file), timestep"

            ds = tg.create_dataset(
                "MultiFrameMoves-Energy", data=mul[:, -1], dtype=np.double
            )
            ds.attrs["Format"] = "lower energy -> higher confidence"

    print("-> results successfully written")


def save_events_parallel(
    options, all_events, max_traxel_id_at, ilp_file, shape, t0, t1, async_=True
):
    processing_pool = Pool()

    for i in range(len(all_events)):
        events = all_events[i]
        first_events = events[0]
        events = events[1:]

        out_dir = options.out_dir.rstrip("/") + "/iter_" + str(i)
        processing_pool.apply_async(
            save_events,
            (
                out_dir,
                copy.copy(events),
                shape,
                t0,
                t1,
                options.label_img_path,
                max_traxel_id_at,
                ilp_file,
                first_events,
            ),
        )

    if async_:
        processing_pool.close()
        processing_pool.join()
    return processing_pool


def save_events(
    out_dir,
    events,
    shape,
    t0,
    t1,
    label_image_path,
    max_traxel_id_at,
    src_filename,
    first_events,
):
    # save events
    print("Saving events...")
    print("Length of events " + str(len(events)))

    if not path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            pass

    working_fns = [out_dir + "/%04d.h5" % timestep for timestep in xrange(t0, t1 + 1)]
    assert len(events) + 1 == len(working_fns)
    with h5py.File(src_filename, "r") as src_file:
        # first timestep without tracking
        with io.LineageH5(working_fns[0], "w") as dest_file:
            print("-- writing empty tracking to base file", working_fns[0])
            # shape = src_file[trans_vector_path].shape
            li_name = label_image_path % (t0, t0 + 1, shape[0], shape[1], shape[2])
            label_img = np.array(src_file[li_name][0, ..., 0]).squeeze()
            seg = dest_file.create_group("segmentation")
            seg.create_dataset("labels", data=label_img, compression="gzip")
            meta = dest_file.create_group("objects/meta")
            ids = np.unique(label_img)
            m = max_traxel_id_at[t0]
            assert np.all(ids == np.arange(m + 1, dtype=ids.dtype))
            ids = ids[ids > 0]
            valid = np.ones(ids.shape)
            meta.create_dataset("id", data=ids, dtype=np.uint32)
            meta.create_dataset("valid", data=valid, dtype=np.uint32)

            # create empty tracking group
            # dest_file.create_group('tracking')
            write_events(first_events, working_fns[0])
            print("-> base file successfully written")

        # tracked timesteps
        for i, events_at in enumerate(events):
            with io.LineageH5(working_fns[i + 1], "w") as dest_file:
                # [t0+1+i,...] or [t0+i,...]?
                # shape = src_file[trans_vector_path].shape
                li_name = label_image_path % (
                    t0 + i + 1,
                    t0 + i + 2,
                    shape[0],
                    shape[1],
                    shape[2],
                )
                label_img = np.array(src_file[li_name][0, ..., 0]).squeeze()
                seg = dest_file.create_group("segmentation")
                seg.create_dataset("labels", data=label_img, compression="gzip")

                meta = dest_file.create_group("objects/meta")
                ids = np.unique(label_img)
                m = max_traxel_id_at[t0 + i + 1]
                assert np.all(ids == np.arange(m + 1, dtype=ids.dtype))
                ids = ids[ids > 0]
                valid = np.ones(ids.shape)
                meta.create_dataset("id", data=ids[::-1], dtype=np.uint32)
                meta.create_dataset("valid", data=valid, dtype=np.uint32)

            write_events(events_at, working_fns[i + 1])


def loadGPClassifier(fn, h5_group="/TransitionGPClassifier/"):
    try:
        from lazyflow.classifiers.gaussianProcessClassifier import (
            GaussianProcessClassifier,
        )
    except:
        raise Exception(
            "cannot import GP Classifier: lazyflow branch gaussianProcessClassifier must be in PYTHONPATH!"
        )

    with h5py.File(fn, "r") as f:
        try:
            g = f[h5_group]
        except:
            raise Exception(h5_group + " does not exist in " + fn)

        gpc = GaussianProcessClassifier()
        gpc.deserialize_hdf5(g)

        features = []
        for op in g["Features"].keys():
            for feat in g["Features"][op]:
                features.append("%s<%s>" % (op, feat))

    return gpc, features


def getH5Dataset(h5group, ds_name):
    if ds_name in h5group.keys():
        return np.array(h5group[ds_name])

    return np.array([])


def train_outlier_svm(options, tracker, ts, fov):
    import pickle
    import trackingfeatures

    print(
        "Storing ground truth labels in hypotheses graph and training an outlier detector SVM from that"
    )
    g = tracker.buildGraph(ts)
    store_label_in_hypotheses_graph(options, g, tracker)
    print("Setting injected solution as active")
    g.set_injected_solution()
    feature_extractor = track.TrackingFeatureExtractor(g, fov)
    print("Train svm")
    feature_extractor.train_track_svm()
    feature_extractor.train_division_svm()
    print("Done training, saving...")
    outlier_track_svm = feature_extractor.get_track_svm()
    outlier_division_svm = feature_extractor.get_division_svm()

    with open(options.save_outlier_svm, "w") as svm_dump:
        pickle.dump(outlier_track_svm, svm_dump)
        pickle.dump(outlier_division_svm, svm_dump)

    print("SVM is trained and stored in " + options.save_outlier_svm)
    options.load_outlier_svm = options.save_outlier_svm

    gt_features_file = options.out_dir.rstrip("/") + "/gt_features.h5"
    feature_extractor.set_track_feature_output_file(gt_features_file)
    feature_extractor.compute_features()

    # create complete lineage trees with extracted features:
    with h5py.File(gt_features_file, "r") as track_features_h5:
        tracks, divisions = trackingfeatures.create_and_link_tracks_and_divisions(
            track_features_h5, ts, getRegionFeatures(ndim)
        )
        lineage_trees = trackingfeatures.build_lineage_trees(tracks, divisions)

    # save lineage trees
    lineage_tree_dump_filename = options.out_dir.rstrip("/") + "/gt_lineage_trees.dump"
    trackingfeatures.save_lineage_dump(
        lineage_tree_dump_filename, tracks, divisions, lineage_trees
    )


def store_label_in_hypotheses_graph(options, graph, tracker):
    # label full graph
    n_it = track.NodeIt(graph)
    a_it = track.ArcIt(graph)
    tmap = graph.getNodeTraxelMap()
    nodeTimestepIdMap = {}
    arcTimestepIdMap = {}

    graph.initLabelingMaps()

    for n in n_it:
        # print(tmap[n].Timestep,tmap[n].Id)
        nodeTimestepIdMap[tmap[n].Timestep, tmap[n].Id] = n
    for a in a_it:
        # print(tmap[g.source(a)].Timestep,tmap[g.source(a)].Id,tmap[g.target(a)].Id)
        arcTimestepIdMap[
            tmap[graph.source(a)].Timestep,
            tmap[graph.source(a)].Id,
            tmap[graph.target(a)].Id,
        ] = a

    # load ground truth from hd5 files and label graph accordingly
    for i in range(graph.earliest_timestep(), graph.latest_timestep() + 1):
        fn = (
            options.gt_pth.rstrip("/")
            + "/"
            + options.gt_path_format_string.format(i)
            + ".h5"
        )
        print("open file:", fn)
        with h5py.File(fn, "r") as f:
            print(f)

            g_tracking = f["tracking"]
            applist = getH5Dataset(g_tracking, "Appearances")
            dislist = getH5Dataset(g_tracking, "Disappearances")
            movlist = getH5Dataset(g_tracking, "Moves")
            spllist = getH5Dataset(g_tracking, "Splits")
            merlist = getH5Dataset(g_tracking, "Mergers")

            mdic = {}
            for m in merlist:
                mdic[m[0]] = np.asscalar(m[1])

            for appset in applist:
                if (i, appset[0]) in nodeTimestepIdMap:
                    graph.addAppearanceLabel(
                        nodeTimestepIdMap[i, appset[0]], mdic.get(appset[0], 1)
                    )
                    graph.addDisappearanceLabel(nodeTimestepIdMap[i, appset[0]], 0)
                else:
                    print("ERROR IN app", i, appset[0])

            for disset in dislist:
                if (i - 1, disset[0]) in nodeTimestepIdMap:
                    graph.addAppearanceLabel(nodeTimestepIdMap[i - 1, disset[0]], 0)
                    graph.addDisappearanceLabel(
                        nodeTimestepIdMap[i - 1, disset[0]], mdic.get(disset[0], 1)
                    )
                else:
                    print("ERROR IN disapp", i - 1, disset[0])

            for movset in movlist:
                if (i - 1, movset[0]) in nodeTimestepIdMap and (
                    i,
                    movset[1],
                ) in nodeTimestepIdMap:
                    graph.addAppearanceLabel(
                        nodeTimestepIdMap[i - 1, movset[0]], mdic.get(movset[0], 1)
                    )
                    graph.addDisappearanceLabel(
                        nodeTimestepIdMap[i, movset[1]], mdic.get(movset[1], 1)
                    )
                    if (i - 1, movset[0], movset[1]) in arcTimestepIdMap:
                        graph.addArcLabel(
                            arcTimestepIdMap[i - 1, movset[0], movset[1]], 1
                        )
                    else:
                        print(
                            "Warning IN move",
                            i - 1,
                            movset[0],
                            movset[1],
                            "  no matching arc found",
                        )
                        # newarc = graph.addArc(nodeTimestepIdMap[i-1,movset[0]],nodeTimestepIdMap[i,movset[0]])
                        # graph.addArcLabel(newarc,1)
                else:
                    print(
                        "ERROR in move ",
                        (i - 1, movset[0]),
                        " or ",
                        (i, movset[1]),
                        "not found",
                    )
                    print(
                        ((i - 1, movset[0]) in nodeTimestepIdMap),
                        ((i, movset[1]) in nodeTimestepIdMap),
                    )

            for splset in spllist:
                if (i - 1, splset[0], splset[1]) in arcTimestepIdMap:
                    graph.addArcLabel(arcTimestepIdMap[i - 1, splset[0], splset[1]], 1)
                else:
                    print("ERROR IN SPLIT arc", i - 1, splset[0], splset[1])
                if (i - 1, splset[0], splset[2]) in arcTimestepIdMap:
                    graph.addArcLabel(arcTimestepIdMap[i - 1, splset[0], splset[2]], 1)
                else:
                    print("ERROR IN SPLIT arc", i - 1, splset[0], splset[2])
                if (
                    (i - 1, splset[0]) in nodeTimestepIdMap
                    and (i, splset[1]) in nodeTimestepIdMap
                    and (i, splset[2]) in nodeTimestepIdMap
                ):
                    graph.addDivisionLabel(nodeTimestepIdMap[i - 1, splset[0]], 1)
                    graph.addDisappearanceLabel(nodeTimestepIdMap[i, splset[1]], 1)
                    graph.addDisappearanceLabel(nodeTimestepIdMap[i, splset[2]], 1)
                    graph.addAppearanceLabel(nodeTimestepIdMap[i - 1, splset[0]], 1)
                else:
                    print("ERROR IN SPLIT node", i - 1, splset[0], splset[1], splset[2])

    return graph


def loadTransClassifier(options):
    if options.trans_fn is None or len(options.trans_fn) == 0:
        trans_classifier = None
    else:
        try:
            from lazyflow.classifiers.TransitionClassifier import TransitionClassifier
        except:
            print("Pythonpath: {}".format(sys.path))
            raise Exception(
                "cannot import Transition Classifier: lazyflow branch gaussianProcessClassifier must be in PYTHONPATH!"
            )

        print("load pre-trained transition classifier")
        gpc, selected_features = loadGPClassifier(
            fn=options.trans_fn, h5_group=options.trans_path
        )
        trans_classifier = TransitionClassifier(gpc, selected_features)

    return trans_classifier


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
        ("RegionAxes", ndim ** 2),
        ("Skewness", 1),
        ("Weighted<PowerSum<0> >", 1),
        ("Coord< Minimum >", ndim),
        ("Coord< Maximum >", ndim),
    ]
    return region_features


def getTraxelStore(options, ilp_fn, time_range, shape):
    max_traxel_id_at = []
    with h5py.File(ilp_fn, "r") as h5file:
        ndim = 3

        print("/".join(options.label_img_path.strip("/").split("/")[:-1]))

        if (
            list(
                h5file[
                    "/".join(options.label_img_path.strip("/").split("/")[:-1])
                ].values()
            )[0].shape[3]
            == 1
        ):
            ndim = 2
        print("ndim=", ndim)

        print(time_range)
        if options.load_traxelstore:
            print("loading traxelstore from file")
            import pickle

            with open(options.load_traxelstore, "rb") as ts_in:
                ts = pickle.load(ts_in)
                fs = pickle.load(ts_in)
                max_traxel_id_at = pickle.load(ts_in)
                ts.set_feature_store(fs)

            info = [int(x) for x in ts.bounding_box()]
            t0, t1 = (info[0], info[4])
            if info[0] != options.mints or (
                options.maxts != -1 and info[4] != options.maxts - 1
            ):
                if options.maxts == -1:
                    options.maxts = info[4] + 1
                print(
                    "Warning: Traxelstore has different time range than requested FOV. Trimming traxels..."
                )
                fov = getFovFromOptions(options, shape, t0, t1)
                fov.set_time_bounds(options.mints, options.maxts - 1)
                new_ts = track.TraxelStore()
                ts.filter_by_fov(new_ts, fov)
                ts = new_ts
        else:
            max_num_mer = int(options.max_num_objects)
            ts, fs, max_traxel_id_at = generate_traxelstore(
                h5file=h5file,
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
                ext_probs=options.ext_probs,
            )

        info = [int(x) for x in ts.bounding_box()]
        t0, t1 = (info[0], info[4])
        print("-> Traxelstore bounding box: " + str(info))

        if options.dump_traxelstore:
            print("dumping traxelstore to file")
            import pickle

            with open(options.dump_traxelstore, "wb") as ts_out:
                pickle.dump(ts, ts_out)
                pickle.dump(fs, ts_out)
                pickle.dump(max_traxel_id_at, ts_out)

    return ts, fs, max_traxel_id_at, ndim, t0, t1


def getFovFromOptions(options, shape, t0, t1):
    [xshape, yshape, zshape] = shape

    fov = track.FieldOfView(
        t0,
        0,
        0,
        0,
        t1,
        options.x_scale * (xshape - 1),
        options.y_scale * (yshape - 1),
        options.z_scale * (zshape - 1),
    )
    return fov


def initializeConservationTracking(options, shape, t0, t1):
    ndim = 2 if shape[-1] == 1 else 3
    rf_fn = "none"
    if options.rf_fn:
        rf_fn = options.rf_fn

    fov = getFovFromOptions(options, shape, t0, t1)
    if ndim == 2:
        [xshape, yshape, zshape] = shape
        assert (
            options.z_scale * (zshape - 1) == 0
        ), "fov of z must be (0,0) if ndim == 2"

    if options.method == "conservation":
        print(">>>>>>>>>>>>>>>>>>>>> Running CPLEX")
        tracker = track.ConsTracking(
            int(options.max_num_objects),
            bool(options.size_dependent_detection_prob),
            options.avg_obj_size[0],
            options.mnd,
            not bool(options.without_divisions),
            options.division_threshold,
            rf_fn,
            fov,
            "none",
            track.ConsTrackingSolverType.CplexSolver,
            ndim,
        )
    elif options.method == "conservation-dynprog":
        print(">>>>>>>>>>>>>>>>>>>>> Running dynprog")
        tracker = track.ConsTracking(
            int(options.max_num_objects),
            bool(options.size_dependent_detection_prob),
            options.avg_obj_size[0],
            options.mnd,
            not bool(options.without_divisions),
            options.division_threshold,
            rf_fn,
            fov,
            "none",
            track.ConsTrackingSolverType.DynProgSolver,
            ndim,
        )
    elif options.method == "conservation-twostage":
        print(">>>>>>>>>>>>>>>>>>>>> Running twostage")
        tracker = track.ConsTracking(
            int(options.max_num_objects),
            bool(options.size_dependent_detection_prob),
            options.avg_obj_size[0],
            options.mnd,
            not bool(options.without_divisions),
            options.division_threshold,
            rf_fn,
            fov,
            "none",
            track.ConsTrackingSolverType.DPInitCplexSolver,
            ndim,
        )
    else:
        raise InvalidArgumentException("Must be conservation or conservation-dynprog")
    return tracker, fov


def getUncertaintyParameter(options, n_iterations=None, sigmas=None):
    sigma_vec = track.VectorOfDouble()

    if sigmas is None:
        for si in options.perturb_sigmas:
            sigma_vec.append(si)
    else:
        for si in sigmas:
            sigma_vec.append(si)

    if options.perturbation_distribution == "DiverseMbest":
        distrib_type = track.DistrId.DiverseMbest
    elif options.perturbation_distribution == "MbestCPLEX":
        distrib_type = track.DistrId.MbestCPLEX
    elif options.perturbation_distribution == "ClassifierUncertainty":
        distrib_type = track.DistrId.ClassifierUncertainty
    elif options.perturbation_distribution == "GaussianPerturbation":
        distrib_type = track.DistrId.GaussianPerturbation
    elif options.perturbation_distribution == "PerturbAndMAP":
        distrib_type = track.DistrId.PerturbAndMAP
    else:
        raise Exception("no such perturbation distribution")

    if n_iterations is None:
        n_iterations = options.num_iterations

    uncertaintyParam = track.UncertaintyParameter(n_iterations, distrib_type, sigma_vec)

    return uncertaintyParam


def prepareFunkeyFiles(options, graph):
    outpath = (
        options.out_dir.rstrip("/")
        + "/"
        + "m_"
        + str(int(options.max_num_objects))
        + "_start"
        + str(graph.earliest_timestep())
        + "_end"
        + str(graph.latest_timestep())
        + "_woc"
        + str(options.woc)
        + "lw"
        + options.funkey_weights.replace(" ", "_")
        + "reg"
        + str(options.funkey_regularizerWeight)
    )

    os.system("mkdir -p " + outpath)

    # remove learning files if present
    os.system("rm -f " + outpath + "/features_0.txt")
    os.system("rm -f " + outpath + "/features.txt")
    os.system("rm -f " + outpath + "/constraints.txt")
    os.system("rm -f " + outpath + "/labels.txt")

    return outpath


def exportFunkeyFiles(options, ts, tracker, trans_classifier):

    g = tracker.buildGraph(ts)
    store_label_in_hypotheses_graph(options, g, tracker)

    # tracker.SetFunkeyExportLabeledGraph(True)

    outpath = prepareFunkeyFiles(options, g)

    features_filename = outpath + "/features.txt"
    constraints_filename = outpath + "/constraints.txt"
    labels_filename = outpath + "/labels.txt"

    if options.only_labels:
        features_filename = ""
        constraints_filename = ""

    tracking_param = tracker.get_conservation_tracking_parameters(
        options.forbidden_cost,
        options.ep_gap,
        not bool(options.without_tracklets),
        options.detection_weight,
        options.division_weight,
        options.transition_weight,
        options.disappearance_cost,
        options.appearance_cost,
        options.with_merger_resolution,
        ndim,
        options.trans_par,
        options.border_width,
        True,  # with_constraints,
        getUncertaintyParameter(options, n_iterations=1),
        options.timeout,
        trans_classifier,
        track.ConsTrackingSolverType.CplexSolver,
    )

    tracker.writeStructuredLearningFiles(
        features_filename, constraints_filename, labels_filename, tracking_param
    )

    return outpath


def learnFunkey(options, tracker, outpath):
    print("calling funkey")

    learnedWeights = tracker.LearnWithFunkey(
        outpath + "/features_0.txt",
        outpath + "/constraints.txt",
        outpath + "/labels.txt",
        options.funkey_weights,
        "--regularizerWeight=" + str(options.funkey_regularizerWeight),
    )

    learnedWeightsName = ["  --det=", "  --div=", "  --tr=", "  --dis=", "  --app="]

    with open(outpath + "/weights.txt", "w") as f:
        for i, w in enumerate(learnedWeights):
            f.write(learnedWeightsName[i] + str(w))

    return learnedWeights


def learn_perturbation_weights(ts, options, shape, trans_classifier, t0, t1):
    print("calling learn_perturbation_weights")
    tracker.SetFunkeyExportLabeledGraph(False)

    optimal_loss = 1e75
    optimal_parameters = []

    os.system("mkdir -p " + outpath + "/pertubation_labels")
    with open(outpath + "/pertubation_labels/pertubation.log", "w") as f:
        for pertubation_parameter in itertools.product(np.arange(-2, 2, 1), repeat=3):
            parameter_name = ""

            pert_tracker, fov = initializeConservationTracking(options, shape, t0, t1)

            pert_tracker.buildGraph(ts)

            sigmas = [0.0, 0.0]
            for si in pertubation_parameter:
                sigmas.append(float(10 ** si))
                parameter_name += "_" + str(float(10 ** si))

            uncertaintyParam = getUncertaintyParameter(options, sigmas=sigmas)

            # save labels of proposals to file:
            pert_tracker.SetFunkeyOutputFiles(
                "",
                "",
                outpath
                + "/pertubation_labels/perturbed_labeling"
                + parameter_name
                + ".txt",
                False,
                uncertaintyParam,
            )
            try:
                pert_tracker.track(
                    options.forbidden_cost,
                    options.ep_gap,
                    not bool(options.without_tracklets),
                    options.detection_weight,
                    options.division_weight,
                    options.transition_weight,
                    options.disappearance_cost,
                    options.appearance_cost,
                    options.with_merger_resolution,
                    ndim,
                    options.trans_par,
                    options.border_width,
                    not bool(options.woconstr),
                    uncertaintyParam,
                    options.timeout,
                    trans_classifier,  # pointer to transition classifier
                )
            except:
                print("ERROR tracker failed to track")

            loss = pert_tracker.HamminglossOfFiles(
                outpath + "/labels.txt",
                outpath
                + "/pertubation_labels/perturbed_labeling"
                + parameter_name
                + "_"
                + str(options.num_iterations - 1)
                + ".txt",
            )
            print(
                "pertubation learning result ",
                str(loss) + "\t" + str(pertubation_parameter),
            )
            f.write(str(loss) + "\t" + str(pertubation_parameter) + "\n")

            if loss < optimal_loss:
                optimal_loss = loss
                optimal_parameters = pertubation_parameter
                print(
                    "found better pertubation parameters:  loss:",
                    optimal_loss,
                    "  para: ",
                    optimal_parameters,
                )
            del pert_tracker

    print(
        "result:   found solution with loss:",
        optimal_loss,
        " with parameters ",
        optimal_parameters,
    )


def runMergerResolving(
    options,
    tracker,
    ts,
    fs,
    hypotheses_graph,
    ilp_fn,
    all_events,
    fov,
    region_features,
    trans_classifier,
    t0,
    with_feature_computation=False,
):
    shape = [int(x) for x in ts.bounding_box()]
    ndim = 2 if shape[-1] == 1 else 3

    with h5py.File(ilp_fn, "r") as h5file:  # open file to access label images
        feature_vectors = []
        scores = []
        for i, event_vector in enumerate(all_events):  # go through all solutions
            if not (options.save_first and i > 0):
                try:
                    os.makedirs(options.out_dir.rstrip("/") + "/iter_" + str(i))
                except:
                    pass

            print("Resolving mergers for solution {}".format(i))
            coordinate_map = track.TimestepIdCoordinateMap()
            coordinate_map.initialize()
            num_mergers = 0
            for timestep, timestep_events in enumerate(event_vector):
                timestep += t0
                mergers = []
                for event in timestep_events:
                    if event.type == track.EventType.Merger:
                        mergers.append(event.traxel_ids[0])
                        traxel = ts.get_traxel(event.traxel_ids[0], timestep)
                        try:
                            extract_coordinates(coordinate_map, h5file, traxel, options)
                        except Exception as e:
                            print(
                                "Error when extracting coordinates for id={} in timestep={}".format(
                                    event.traxel_ids[0], timestep
                                )
                            )
                            print(type(e))
                            print(e)
                            raise Exception
                        num_mergers += 1
                for event in timestep_events:
                    if (
                        event.type == track.EventType.Move
                        and event.traxel_idx[1] in mergers
                    ):
                        traxel = ts.get_traxel(event.traxel_ids[0], timestep - 1)
                        try:
                            extract_coordinates(coordinate_map, h5file, traxel, options)
                        except Exception as e:
                            print(
                                "Error when extracting coordinates for id={} in timestep={}".format(
                                    event.traxel_ids[0], timestep - 1
                                )
                            )

            print(
                "Found {} merger events in proposal {}. Resolving them...".format(
                    num_mergers, i
                )
            )

            hypotheses_graph.set_solution(i)

            resolved_events = tracker.resolve_mergers(
                event_vector,
                coordinate_map.get(),
                float(options.ep_gap),
                options.transition_weight,
                not bool(options.without_tracklets),
                ndim,
                options.trans_par,
                not bool(options.woconstr),
                trans_classifier,
            )

            with h5py.File(options.raw_filename, "r") as raw_h5:
                print(
                    "Update labelimage and compute new features after merger resolving "
                    "for solution {}".format(i)
                )
                pb = ProgressBar(0, len(resolved_events))
                for timestep, timestep_events in enumerate(resolved_events):
                    pb.show()
                    timestep += t0
                    for event in timestep_events:
                        if event.type == track.EventType.ResolvedTo:
                            merger_traxel = ts.get_traxel(event.traxel_ids[0], timestep)
                            new_traxel_ids = event.traxel_ids[1:]
                            for new_id in new_traxel_ids:
                                try:
                                    ts.get_traxel(new_id, timestep)
                                    # print("MR: Found traxel {} at timestep {} in traxelstore.".format(new_id, timestep))
                                except:
                                    # print("MR: Could not find traxel {} at timestep {} in traxelstore, adding it now.".format(new_id, timestep))
                                    traxel = track.Traxel()
                                    traxel.set_feature_store(fs)
                                    # traxel.set_x_scale(x_scale)
                                    # traxel.set_y_scale(y_scale)
                                    # traxel.set_z_scale(z_scale)
                                    traxel.Id = int(new_id)
                                    traxel.Timestep = int(timestep)
                                    ts.add(fs, traxel)
                            update_merger_features(
                                coordinate_map,
                                h5file,
                                merger_traxel,
                                new_traxel_ids,
                                raw_h5,
                                options,
                                fs,
                                ts,
                                timestep,
                            )
            resolved_graph = tracker.get_resolved_hypotheses_graph()
            # extract features of this solution

            if with_feature_computation:
                feature_vector, score, feature_names = extract_features_and_compute_score(
                    options.reranker_weight_file,
                    options.load_outlier_svm,
                    options.out_dir,
                    i,
                    resolved_graph,
                    ts,
                    fov,
                    None,
                    region_features,
                )
                feature_vectors.append(feature_vector)
                scores.append(score)

        if with_feature_computation:
            print("Storing feature vectors of all solutions")
            np.savetxt(
                options.out_dir.rstrip("/") + "/feature_vectors.txt",
                np.array(feature_vectors).transpose(),
            )
            print("Storing feature names")
            with open(
                options.out_dir.rstrip("/") + "/feature_names.txt", "w"
            ) as feature_names_file:
                for f in feature_names:
                    feature_names_file.write(f + "\n")
    return scores


if __name__ == "__main__":
    options, args = getConfigAndCommandLineArguments()

    # get filenames
    numArgs = len(args)
    fns = []
    if numArgs > 0:
        for arg in args:
            print(arg)
            fns.extend(glob.glob(arg))
        fns.sort()
        print(fns)

    print(fns)
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

    trans_classifier = loadTransClassifier(options)

    # set average object size if chosen
    obj_size = [0]
    if options.avg_obj_size != 0:
        obj_size[0] = options.avg_obj_size
    else:
        options.avg_obj_size = obj_size

    # find shape of dataset
    with h5py.File(ilp_fn, "r") as h5file:
        shape = list(h5file["/".join(options.label_img_path.split("/")[:-1])].values())[
            0
        ].shape[1:4]

    # read all traxels into TraxelStore
    ts, fs, max_traxel_id_at, ndim, t0, t1 = getTraxelStore(
        options, ilp_fn, time_range, shape
    )

    print("Start tracking...")
    if (
        options.method != "conservation"
        and options.method != "conservation-dynprog"
        and options.method != "conservation-twostage"
    ):
        raise Exception("unknown tracking method: " + options.method)

    w_det, w_div, w_trans, w_dis, w_app, = (
        options.detection_weight,
        options.division_weight,
        options.transition_weight,
        options.disappearance_cost,
        options.appearance_cost,
    )

    # generate structured learning files
    if options.export_funkey and len(options.gt_pth) > 0:
        tracker, fov = initializeConservationTracking(options, shape, t0, t1)
        outpath = exportFunkeyFiles(options, ts, tracker, trans_classifier)

        if options.only_labels:
            print("finished writing labels to " + outpath)
            exit(0)

        if options.learn_funkey:
            learned_weights = learnFunkey(options, tracker, outpath)
            w_det, w_div, w_trans, w_dis, w_app, = learned_weights

        if options.learn_perturbation_weights:
            learn_perturbation_weights(ts, options, shape, trans_classifier, t0, t1)
            exit(0)

    # -------------------------------------------------------
    # perform the real tracking
    tracker, fov = initializeConservationTracking(options, shape, t0, t1)

    # train outlier svm if needed
    if len(options.save_outlier_svm) > 0 and len(options.gt_pth) > 0:
        train_outlier_svm(options, tracker, ts, fov)

    if options.num_iterations == 0:
        exit(0)

    # build hypotheses graph
    print("tracking with weights ", w_det, w_div, w_trans, w_dis, w_app)
    hypotheses_graph = tracker.buildGraph(ts, options.max_nearest_neighbors)

    if options.hypotheses_graph_filename:
        import pickle

        with open(options.hypotheses_graph_filename, "wb") as hg_out:
            pickle.dump(hypotheses_graph, hg_out)
            pickle.dump(fov, hg_out)
            pickle.dump(fs, hg_out)
        sys.exit(0)

    if options.w_labeling:
        assert (
            len(options.gt_pth) > 0
        ), "if labeling should be loaded, please provide a path to the ground truth in --gt-pth"
        store_label_in_hypotheses_graph(options, hypotheses_graph, tracker)

    # perturbation settings
    uncertaintyParam = getUncertaintyParameter(options)
    if options.num_iterations == 0 or options.num_iterations == 1 or options.save_first:
        proposal_out = ""
    else:
        proposal_out = options.out_dir.rstrip("/") + "/proposal_labeling"

    tracker.setTrackLabelingExportFile(proposal_out)

    solver = track.ConsTrackingSolverType.CplexSolver
    if options.method == "conservation-dynprog":
        solver = track.ConsTrackingSolverType.DynProgSolver
    elif options.method == "conservation-twostage":
        solver = track.ConsTrackingSolverType.DPInitCplexSolver

    params = tracker.get_conservation_tracking_parameters(
        options.forbidden_cost,
        options.ep_gap,
        not bool(options.without_tracklets),
        w_det,
        w_div,
        w_trans,
        w_dis,
        w_app,
        options.with_merger_resolution,
        ndim,
        options.trans_par,
        options.border_width,
        not bool(options.woconstr),
        uncertaintyParam,
        options.timeout,
        trans_classifier,  # pointer to transition classifier
        solver,  # Solver
        False,  # training to hard constraints
        options.num_threads,
    )

    if options.motionModelWeight > 0:
        print(
            "Registering motion model with weight {}".format(options.motionModelWeight)
        )
        params.register_motion_model4_func(
            swirl_motion_func_creator(options.motionModelWeight),
            options.motionModelWeight * 25.0,
        )

    # dynprog settings
    if options.max_num_paths is not None:
        params.max_number_paths = options.max_num_paths
    params.with_swap = not options.without_swaps

    if not options.without_tracklets:
        traxel_graph = hypotheses_graph
        hypotheses_graph = traxel_graph.generate_tracklet_graph()

    # tracker.plot_hypotheses_graph(
    #     hypotheses_graph,
    #     "/Users/chaubold/Desktop/multitrack-graph.dot",
    #     not bool(options.without_tracklets),
    #     not bool(options.without_divisions),
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     options.trans_par,
    #     options.border_width
    #     )

    # track!
    all_events = tracker.track(params, False)
    # all_events = tracker.track(
    #     options.forbidden_cost,
    #     options.ep_gap,
    #     not bool(options.without_tracklets),
    #     w_det,
    #     w_div,
    #     w_trans,
    #     w_dis,
    #     w_app,
    #     options.with_merger_resolution,
    #     ndim,
    #     options.trans_par,
    #     options.border_width,
    #     not bool(options.woconstr),
    #     uncertaintyParam,
    #     options.timeout,
    #     trans_classifier, # pointer to transition classifier
    #     options.num_threads
    # )

    tracker.setTrackLabelingExportFile("")

    # # dispatch saving events already while merger resolving runs
    # if not options.skip_saving and not options.save_first:
    #     parallel_save_process_pool = save_events_parallel(options, all_events, max_traxel_id_at, ilp_fn, t0, t1, False)

    # run merger resolving
    if (
        options.with_merger_resolution
        and options.max_num_objects > 1
        and len(options.raw_filename) > 0
    ):
        region_features = getRegionFeatures(ndim)
        try:
            runMergerResolving(
                options,
                tracker,
                ts,
                fs,
                hypotheses_graph,
                ilp_fn,
                all_events,
                fov,
                region_features,
                trans_classifier,
                t0,
                True,
            )
        except BaseException as e:
            print("WARNING: Merger Resolving crashed...: {}".format(e))

    stop = time.time()
    since = stop - start

    if options.w_labeling:
        hypotheses_graph.write_hypotheses_graph_state(
            options.out_dir.rstrip("/") + "/results_all.txt"
        )

    # save
    if options.save_first:
        events = all_events[0]
        first_events = events[0]
        events = events[1:]
        out_dir = options.out_dir.rstrip("/") + "/iter_0"
        save_events(
            out_dir,
            events,
            shape,
            t0,
            t1,
            options.label_img_path,
            max_traxel_id_at,
            ilp_fn,
            first_events,
        )
    elif not options.skip_saving:
        # parallel_save_process_pool.close()
        # parallel_save_process_pool.join()
        save_events_parallel(
            options, all_events, max_traxel_id_at, ilp_fn, shape, t0, t1, True
        )

    print("Elapsed time [s]: " + str(int(since)))
    print("Elapsed time [min]: " + str(int(since) / 60))
    print("Elapsed time [h]: " + str(int(since) / 3600))
