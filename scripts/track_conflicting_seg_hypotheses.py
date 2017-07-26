"""
Take a set of label images that represent different segmentation hypotheses,
track them, and create a final result by cherry-picking the segments that were part of the tracking solution
"""

# pythonpath modification to make hytra available
# for import without requiring it to be installed
from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import os
import sys
import gzip
import pickle
import numpy as np
import logging
from skimage.external import tifffile
import vigra
try:
    import commentjson as json
except ImportError:
    import json
import configargparse as argparse
sys.path.insert(0, os.path.abspath('..'))
from hytra.core.ilastikhypothesesgraph import IlastikHypothesesGraph
from hytra.core.fieldofview import FieldOfView
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
from hytra.core.jsongraph import writeToFormattedJSON
import hytra.jst.conflictingsegmentsprobabilitygenerator as probabilitygenerator
import hytra.jst.classifiertrainingexampleextractor
from hytra.core.ilastik_project_options import IlastikProjectOptions

def getLogger():
    return logging.getLogger("track_conflicting_seg_hypotheses")

def constructFov(shape, t0, t1, scale=[1, 1, 1]):
    [xshape, yshape, zshape] = shape
    [xscale, yscale, zscale] = scale

    fov = FieldOfView(t0, 0, 0, 0, t1, xscale * (xshape - 1), yscale * (yshape - 1),
                      zscale * (zshape - 1))
    return fov

def remap_label_image(label_images, mapping):
    """ 
    given a label image and a mapping, creates and 
    returns a new label image with remapped object pixel values 
    """
    remapped_label_image = np.zeros(list(label_images.values())[0].shape, dtype=list(label_images.values())[0].dtype)
    for origObject, trackId in mapping.items():
        objectId, filename = origObject
        remapped_label_image[label_images[filename] == objectId] = trackId

    return remapped_label_image

def save_frame_to_tif(timestep, label_image, options):
    if options.is_groundtruth:
        filename = options.output_dir + '/man_track' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    else:
        filename = options.output_dir + '/mask' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    # label_image = np.swapaxes(label_image, 0, 1)
    if len(label_image.shape) == 2: # 2d
        vigra.impex.writeImage(label_image.astype('uint16'), filename)
    else: # 3D
        label_image = np.transpose(label_image, axes=[2, 0, 1])
        tifffile.imsave(filename, label_image.astype('uint16'))


def save_tracks(tracks, options):
    """
    Take a dictionary indexed by TrackId which contains
    a list [parent, begin, end] per track, and save it 
    in the text format of the cell tracking challenge.
    """
    if options.is_groundtruth:
        filename = options.output_dir + '/man_track.txt'
    else:
        filename = options.output_dir + '/res_track.txt'
    with open(filename, 'wt') as f:
        for key, value in tracks.items():
            if key ==  None:
                continue
            # our track value contains parent, begin, end
            # but here we need begin, end, parent. so swap
            f.write("{} {} {} {}\n".format(key, value[1], value[2], value[0]))

def mapGroundTruth(options, hypotheses_graph, trackingGraph, probGenerator):
    """
    If we were given a ground truth, we can map it to the graph and either train new classifiers,
    or run structured learning to find the optimal weights. 
    """
    getLogger().info("Map ground truth")
    jsonGT = probGenerator.findGroundTruthJaccardScoreAndMapping(
        hypotheses_graph,
        options.gt_label_image_file,
        options.gt_label_image_path,
        options.gt_text_file,
        options.gt_jaccard_threshold
    )

    if options.out_obj_count_classifier_file is not None and options.out_obj_count_classifier_path is not None:
        getLogger().info("Training Random Forest detection classifier")
        rf = hytra.jst.classifiertrainingexampleextractor.trainDetectionClassifier(hypotheses_graph, jsonGT, numSamples=100)
        rf.save(options.out_obj_count_classifier_file, options.out_obj_count_classifier_path)

        getLogger().info("Quitting, because you probably want to set up a new graph using the new classifiers...")
        sys.exit(0)

    try:
        import multiHypoTracking_with_cplex as mht
    except ImportError:
        try:
            import multiHypoTracking_with_gurobi as mht
        except ImportError:
            pass
    if mht:
        getLogger().info("Learn weights")
        weights = mht.train(trackingGraph.model, jsonGT)

        if options.learned_weights_json_filename is not None:
            writeToFormattedJSON(options.learned_weights_json_filename, weights)
        return weights
    return None

def setupGraph(options):
    """
    Configure where to load raw data and classifiers from, then first find all objects and their features 
    and probabilities with the probabilitygenerator, and finally build a hypotheses graph and prepare it for tracking
    """
    ilpOptions = IlastikProjectOptions()
    ilpOptions.labelImagePath = options.label_image_paths[0]
    ilpOptions.labelImageFilename = options.label_image_files[0]

    ilpOptions.rawImagePath = options.raw_data_path
    ilpOptions.rawImageFilename = options.raw_data_file
    ilpOptions.rawImageAxes = options.raw_data_axes
    
    ilpOptions.sizeFilter = [10, 100000]
    ilpOptions.objectCountClassifierFilename = options.obj_count_classifier_file
    ilpOptions.objectCountClassifierPath = options.obj_count_classifier_path
    
    withDivisions = options.with_divisions
    if withDivisions:
        ilpOptions.divisionClassifierFilename = options.div_classifier_file
        ilpOptions.divisionClassifierPath = options.div_classifier_path
    else:
        ilpOptions.divisionClassifierFilename = None

    getLogger().info("Extracting traxels from images")
    probGenerator = probabilitygenerator.ConflictingSegmentsProbabilityGenerator(
        ilpOptions, 
        options.label_image_files[1:],
        options.label_image_paths[1:],
        pluginPaths=['../hytra/plugins'],
        useMultiprocessing=not options.disableMultiprocessing)

    # restrict range of timeframes used for learning and tracking
    if options.end_frame < 0:
        options.end_frame += probGenerator.timeRange[1] + 1
    assert(options.init_frame < probGenerator.timeRange[1])
    assert(options.end_frame <= probGenerator.timeRange[1])
    probGenerator.timeRange = (options.init_frame, options.end_frame)

    probGenerator.fillTraxels(usePgmlink=False)
    fieldOfView = constructFov(probGenerator.shape,
                            probGenerator.timeRange[0],
                            probGenerator.timeRange[1],
                            [probGenerator.x_scale,
                            probGenerator.y_scale,
                            probGenerator.z_scale])

    getLogger().info("Building hypotheses graph")
    hypotheses_graph = IlastikHypothesesGraph(
        probabilityGenerator=probGenerator,
        timeRange=probGenerator.timeRange,
        maxNumObjects=1,
        numNearestNeighbors=options.max_nearest_neighbors,
        fieldOfView=fieldOfView,
        withDivisions=withDivisions,
        divisionThreshold=0.1,
        maxNeighborDistance=options.max_neighbor_distance
    )

    # if options.with_tracklets:
    #     hypotheses_graph = hypotheses_graph.generateTrackletGraph()

    getLogger().info("Preparing for tracking")
    hypotheses_graph.insertEnergies()
    trackingGraph = hypotheses_graph.toTrackingGraph()
    
    if options.do_convexify or options.use_flow_solver:
        getLogger().info("Convexifying graph energies...")
        trackingGraph.convexifyCosts()

    if options.graph_json_filename is not None:
        writeToFormattedJSON(options.graph_json_filename, trackingGraph.model)

    return fieldOfView, hypotheses_graph, ilpOptions, probGenerator, trackingGraph

def runTracking(options, trackingGraph, weights=None):
    """
    Track the given graph with the given weights, if None the weights will be loaded from a json file.
    **Returns** the tracking result dictionary
    """

    getLogger().info("Run tracking...")
    if weights is None:
        getLogger().info("Loading weights from " + options.weight_json_filename)
        with open(options.weight_json_filename, 'r') as f:
            weights = json.load(f)

        # if withDivisions:
        #     weights = {"weights" : [10, 10, 10, 500, 500]}
        # else:
        #     weights = {"weights" : [10, 10, 500, 500]}
    else:
        getLogger().info("Using learned weights!")

    if options.use_flow_solver:
        import dpct
        result = dpct.trackFlowBased(trackingGraph.model, weights)
    else:
        try:
            import multiHypoTracking_with_cplex as mht
        except ImportError:
            try:
                import multiHypoTracking_with_gurobi as mht
            except ImportError:
                raise ImportError("No version of ILP solver found")
        result = mht.track(trackingGraph.model, weights)
    
    if options.result_json_filename is not None:
        writeToFormattedJSON(options.result_json_filename, result)

    return result

def run_pipeline(options):
    """
    Run the complete tracking pipeline with competing segmentation hypotheses
    """

    # set up probabilitygenerator (aka traxelstore) and hypothesesgraph or load them from a dump
    if options.load_graph_filename is not None:
        getLogger().info("Loading state from file: " + options.load_graph_filename)
        with gzip.open(options.load_graph_filename, 'r') as graphDump:
            ilpOptions = pickle.load(graphDump)
            probGenerator = pickle.load(graphDump)
            fieldOfView = pickle.load(graphDump)
            hypotheses_graph = pickle.load(graphDump)
            trackingGraph = pickle.load(graphDump)
        getLogger().info("Done loading state from file")
    else:
        fieldOfView, hypotheses_graph, ilpOptions, probGenerator, trackingGraph = setupGraph(options)
        
        if options.dump_graph_filename is not None:
            getLogger().info("Saving state to file: " + options.dump_graph_filename)
            with gzip.open(options.dump_graph_filename, 'w') as graphDump:
                pickle.dump(ilpOptions, graphDump)
                pickle.dump(probGenerator, graphDump)
                pickle.dump(fieldOfView, graphDump)
                pickle.dump(hypotheses_graph, graphDump)
                pickle.dump(trackingGraph, graphDump)
            getLogger().info("Done saving state to file")

    # map groundtruth to hypothesesgraph if all required variables are specified
    weights = None
    if options.gt_label_image_file is not None and options.gt_label_image_path is not None \
        and options.gt_text_file is not None and options.gt_jaccard_threshold is not None:
        weights = mapGroundTruth(options, hypotheses_graph, trackingGraph, probGenerator)

    # track
    result = runTracking(options, trackingGraph, weights)

    # insert the solution into the hypotheses graph and from that deduce the lineages
    getLogger().info("Inserting solution into graph")
    hypotheses_graph.insertSolution(result)
    hypotheses_graph.computeLineage()

    mappings = {} # dictionary over timeframes, containing another dict objectId -> trackId per frame
    tracks = {} # stores a list of timeframes per track, so that we can find from<->to per track
    trackParents = {} # store the parent trackID of a track if known

    for n in hypotheses_graph.nodeIterator():
        frameMapping = mappings.setdefault(n[0], {})
        if 'trackId' not in hypotheses_graph._graph.node[n]:
            raise ValueError("You need to compute the Lineage of every node before accessing the trackId!")
        trackId = hypotheses_graph._graph.node[n]['trackId']
        traxel = hypotheses_graph._graph.node[n]['traxel']
        if trackId is not None:
            frameMapping[(traxel.idInSegmentation, traxel.segmentationFilename)] = trackId
        if trackId in tracks:
            tracks[trackId].append(n[0])
        else:
            tracks[trackId] = [n[0]]
        if 'parent' in hypotheses_graph._graph.node[n]:
            assert(trackId not in trackParents)
            trackParents[trackId] = hypotheses_graph._graph.node[hypotheses_graph._graph.node[n]['parent']]['trackId']

    # write res_track.txt
    getLogger().info("Writing track text file")
    trackDict = {}
    for trackId, timestepList in tracks.items():
        timestepList.sort()
        try:
            parent = trackParents[trackId]
        except KeyError:
            parent = 0
        trackDict[trackId] = [parent, min(timestepList), max(timestepList)]
    save_tracks(trackDict, options) 

    # export results
    getLogger().info("Saving relabeled images")
    pluginManager = TrackingPluginManager(verbose=options.verbose, pluginPaths=options.pluginPaths)
    pluginManager.setImageProvider('LocalImageLoader')
    imageProvider = pluginManager.getImageProvider()
    timeRange = probGenerator.timeRange

    for timeframe in range(timeRange[0], timeRange[1]):
        label_images = {}
        for f, p in zip(options.label_image_files, options.label_image_paths):
            label_images[f] = imageProvider.getLabelImageForFrame(f, p, timeframe)
        remapped_label_image = remap_label_image(label_images, mappings[timeframe])
        save_frame_to_tif(timeframe, remapped_label_image, options)


if __name__ == "__main__":
    class Formatter( argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter): 
        pass

    parser = argparse.ArgumentParser(
        description='''Multi-Segmentation-Hypotheses Tracking Pipeline.
        
        ============================
        General usage
        ============================
        
        This pipeline can take not only one segmentation, but several label images, finds conflicting hypotheses,
        builds a tracking graph for all of them, tracks them, and constructs a result in the Cell Tracking Challenge format.

        Raw data has to be provided as a single HDF5 volume, and can be configured with the parameters
        --raw-data-file, --raw-data-path and --raw-data-axes.

        To specify several segmentation label images, add the --label-image-file option several times, everytime adding another label image file.
        Label images are required to have the old ilastik-internal style where each time frame is saved as different group.
        Use the "ctc/segmentation_to_hdf5.py" script to convert a series of tiff files into that format, or
        "segmentation_to_labelimage.py" to convert a single HDF5 label volume.

        Tracking:
        ---------

        For tracking, the pipeline needs an object-count-classifier and, if divisions are present a division-classifier (then also pass "--with-divisions"!). 
        Optionally also a transition-classifier, otherwise euclidean center distance will be used.
        You have to pass in the respective filenames and paths inside the HDF5 files.

        Tracking can be run either with an ILP solver (multiHypothesesTracking must be available), or with the flow based solver.
        To use the flow-solver, pass "--use-flow-solver". Additionally, you can influence in which spatial neighborhood a transition may happen (--max-neighbor-distance),
        and how many links are added (--max-nearest-neighbors). Remember that this number refers to the neighbors either in forward or backwards direction, and the graph will contain
        the union of both, so probably edges more than this number per node.         

        Tracking needs weights (to configure the importance of the individual classifiers etc), these can be loaded from a JSON encoded file (--weight-json-file),
        or learned (see the Ground Truth section below).
        
        The result will be given in the cell tracking challenge format, so you only need to configure the folder where to put this result (--ctc-output-dir).

        ============================
        Training from a Ground Truth
        ============================

        To learn the weight of the respective classifiers using structured learning, you can pass in a ground truth (GT) solution. 
        The ground truth is given as a cell tracking challenge text file (--gt-text-file), and an ilastik-style label image of the GT segmentation 
        (e.g. convert the CTC GT with "ctc/segmentation_to_hdf5.py") --gt-label-image-*. 
        Additionally you can specify the minimum jaccard score of any segmentation hypotheses with a GT object to be considered as "matching" (--gt-jaccard-threshold).

        You can save the learned weights to a file by specifying "-learned-weight-json-file".

        If you want to train an object count classifier given the ground truth INSTEAD of running structured learning, specify the "--out-object-count-classifier*" parameters. 
        The script will stop after training, because you probably want to pass in this newly trained classifier into "--object-count-classifier-*" for tracking and structured learning. 
        ''',
        formatter_class=Formatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file', required=False)

    parser.add_argument("--do-convexify", dest='do_convexify', action='store_true', default=False)
    parser.add_argument('--plugin-paths', dest='pluginPaths', type=str, nargs='+',
                        default=[os.path.abspath('../hytra/plugins')],
                        help='A list of paths to search for plugins for the tracking pipeline.')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)
    parser.add_argument('--disable-multiprocessing', dest='disableMultiprocessing', action='store_true',
                        help='Do not use multiprocessing to speed up computation',
                        default=False)

    # Raw Data:
    group = parser.add_argument_group('Input Images', 'Raw data and label images')
    group.add_argument('--raw-data-file', type=str, dest='raw_data_file', default=None,
                      help='filename to the raw h5 file')
    group.add_argument('--raw-data-path', type=str, dest='raw_data_path', default='exported_data',
                      help='Path inside the raw h5 file to the data')
    group.add_argument("--raw-data-axes", dest='raw_data_axes', type=str, default='txyzc',
                        help="axes ordering of the produced raw image, e.g. xyztc.")

    # Label images:
    group.add_argument('--label-image-file', type=str, dest='label_image_files', action='append',
                      help='Label image filenames of the different segmentation hypotheses')
    group.add_argument('--label-image-path', dest='label_image_paths', type=str, action='append',
                        help='''internal hdf5 path to label image. If only exactly one argument is given, it will be used for all label images,
                        otherwise there need to be as many label image paths as filenames.
                        Defaults to "/TrackingFeatureExtraction/LabelImage/0000/[[%%d, 0, 0, 0, 0], [%%d, %%d, %%d, %%d, 1]]" for all images if not specified''')
    
    # Ground Truth (if available):
    group = parser.add_argument_group('Ground Truth', 'Specify a ground truth segmentation and tracking which will be used for learning')
    group.add_argument('--gt-label-image-file', type=str, dest='gt_label_image_file', default=None,
                      help='Ground Truth Label image filename')
    group.add_argument('--gt-label-image-path', dest='gt_label_image_path', type=str,
                        help="internal hdf5 path to gt label image.",
                        default="/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]")
    group.add_argument('--gt-text-file', type=str, dest='gt_text_file', default=None,
                      help='Ground Truth text path+filename (/path/to/res_track.txt)')
    group.add_argument('--gt-jaccard-threshold', type=float, dest='gt_jaccard_threshold', default=0.3,
                      help='Minimum jaccard score of a segmentation hypotheses with the GT to be used as a match')

    # learned classifiers:
    group = parser.add_argument_group('Classifier Output', 'Where to save learned classifiers')
    group.add_argument('--out-object-count-classifier-path', dest='out_obj_count_classifier_path', type=str,
                        default='/CountClassification/',
                        help='internal hdf5 path to object count probabilities')
    group.add_argument('--out-object-count-classifier-file', dest='out_obj_count_classifier_file', type=str,
                        help='Filename of the HDF file where the new object count classifier will be stored.')

    # classifiers:
    group = parser.add_argument_group('Classifier Input', 'Where to load classifiers from')
    group.add_argument('--object-count-classifier-path', dest='obj_count_classifier_path', type=str,
                        default='/CountClassification/Probabilities/0/',
                        help='internal hdf5 path to object count probabilities')
    group.add_argument('--object-count-classifier-file', dest='obj_count_classifier_file', type=str, required=True,
                        help='Filename of the HDF file containing the object count classifier.')
    group.add_argument('--division-classifier-path', dest='div_classifier_path', type=str, default='/DivisionDetection/Probabilities/0/',
                        help='internal hdf5 path to division probabilities')
    group.add_argument('--division-classifier-file', dest='div_classifier_file', type=str, default=None,
                        help='Filename of the HDF file containing the division classifier.')
    group.add_argument('--transition-classifier-file', dest='transition_classifier_filename', type=str,
                        default=None)
    group.add_argument('--transition-classifier-path', dest='transition_classifier_path', type=str, default='/')

    # Intermediate output
    group = parser.add_argument_group('Intermediate outputs', 'Provides export for JSON file graph, result, and weights, as well as graph dumping(slow!)')
    group.add_argument('--graph-json-file', type=str, dest='graph_json_filename', default=None,
                        help='filename where to save the generated JSON graph to')
    group.add_argument('--result-json-file', type=str, dest='result_json_filename', default=None,
                        help='filename where to save the results to in JSON format')
    group.add_argument('--learned-weight-json-file', type=str, dest='learned_weights_json_filename', default=None,
                        help='filename where to save the weights to in JSON format')

    # pickle graph in between so we don't have to recompute features etc?
    group.add_argument('--dump-graph-to', type=str, dest='dump_graph_filename', default=None,
                        help='filename where to dump the graph etc to')
    group.add_argument('--load-graph-from', type=str, dest='load_graph_filename', default=None,
                        help='filename where to load a dumped graph from')

    # Tracking:
    group = parser.add_argument_group('Graph and Tracking', 'Parameters that configure hypotheses graph construction and tracking')
    group.add_argument('--weight-json-file', type=str, dest='weight_json_filename', default=None,
                        help='filename where to load the weights from JSON - if no GT is given')
    group.add_argument("--init-frame", default=0, type=int, dest='init_frame',
                        help="where to begin reading the frames")
    group.add_argument("--end-frame", default=-1, type=int, dest='end_frame',
                        help="where to end frames")
    group.add_argument('--max-neighbor-distance', dest='max_neighbor_distance', type=float, default=200)
    group.add_argument('--max-nearest-neighbors', dest='max_nearest_neighbors', type=int, default=4)
    group.add_argument('--with-divisions', dest='with_divisions', action='store_true')
    group.add_argument("--use-flow-solver", dest='use_flow_solver', action='store_true',
                        help='Switch to non-optimal solver instead of ILP solver')

    # Output
    group = parser.add_argument_group('Output', 'Result files')
    group.add_argument('--ctc-output-dir', type=str, dest='output_dir', default=None, required=True,
                        help='foldername in which all the output is stored')
    group.add_argument('--is-gt', dest='is_groundtruth', action='store_true')
    group.add_argument('--ctc-filename-zero-pad-length', type=int, dest='filename_zero_padding', default=3)

    # parse command line
    options, unknown = parser.parse_known_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    # assert(len(options.label_image_files) > 1)
    if options.label_image_paths is None or len(options.label_image_paths) == 0:
        options.label_image_paths = ['/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]']
    if len(options.label_image_paths) < len(options.label_image_files) and len(options.label_image_paths) == 1:
        options.label_image_paths = options.label_image_paths * len(options.label_image_files)   
    assert(len(options.label_image_paths) == len(options.label_image_files) )

    # make sure output directory exists
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    run_pipeline(options)
