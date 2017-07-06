# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import configargparse as argparse
import numpy as np
import h5py
import vigra
import time
import logging
from skimage.external import tifffile
from hytra.core.jsongraph import JsonTrackingGraph
from hytra.pluginsystem.plugin_manager import TrackingPluginManager

def getLogger():
    return logging.getLogger(__name__)

def save_frame_to_tif(timestep, label_image, options):
    if len(options.label_image_filename) == 1:
        filename = options.output_dir + '/man_track' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    else:
        filename = options.output_dir + '/mask' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    label_image = np.swapaxes(label_image, 0, 1)
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
    if options.is_ground_truth:
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


def remap_label_image(label_image, mapping):
    """ 
    given a label image and a mapping, creates and 
    returns a new label image with remapped object pixel values 
    """
    remapped_label_image = np.zeros(label_image.shape, dtype=label_image.dtype)
    for dest, src in mapping.items():
        remapped_label_image[label_image == dest] = src

    return remapped_label_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert H5 event tracking solution to CTC format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--ctc-output-dir', type=str, dest='output_dir', required=True,
                        help='Folder where to save the label images starting with res_track00.tif, as well as a file res_track.txt')
    parser.add_argument('--ctc-filename-zero-pad-length', type=int, dest='filename_zero_padding', default='3')
    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--result-json-file', required=True, type=str, dest='result_filename',
                        help='Filename of the json file containing results')
    parser.add_argument('--label-image-file', required=True, type=str, dest='label_image_filename',
                        help='Filename of the ilastik-style segmentation HDF5 file')
    parser.add_argument('--label-image-path', dest='label_image_path', type=str,
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]',
                        help='internal hdf5 path to label image')
    parser.add_argument('--plugin-paths', dest='pluginPaths', type=str, nargs='+',
                        default=[os.path.abspath('../../hytra/plugins'), os.path.abspath('../hytra/plugins')],
                        help='A list of paths to search for plugins for the tracking pipeline.')
    parser.add_argument("--is-ground-truth", dest='is_ground_truth', action='store_true', default=False)
    parser.add_argument('--links-to-num-next-frames', dest='linksToNumNextFrames', type=int, default=1)

    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    args, unknown = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    getLogger().debug("Ignoring unknown parameters: {}".format(unknown))

    # make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load graph and compute lineages
    getLogger().debug("Loading graph and result")
    trackingGraph = JsonTrackingGraph(model_filename=args.model_filename, result_filename=args.result_filename)
    hypothesesGraph = trackingGraph.toHypothesesGraph()
    hypothesesGraph.computeLineage(1, 1, args.linksToNumNextFrames)

    mappings = {} # dictionary over timeframes, containing another dict objectId -> trackId per frame
    tracks = {} # stores a list of timeframes per track, so that we can find from<->to per track
    trackParents = {} # store the parent trackID of a track if known
    gapTrackParents = {}

    for n in hypothesesGraph.nodeIterator():
        frameMapping = mappings.setdefault(n[0], {})
        if 'trackId' not in hypothesesGraph._graph.node[n]:
            raise ValueError("You need to compute the Lineage of every node before accessing the trackId!")
        trackId = hypothesesGraph._graph.node[n]['trackId']
        if trackId is not None:
            frameMapping[n[1]] = trackId
        if trackId in tracks.keys():
            tracks[trackId].append(n[0])
        else:
            tracks[trackId] = [n[0]]
        if 'parent' in hypothesesGraph._graph.node[n]:
            assert(trackId not in trackParents)
            trackParents[trackId] = hypothesesGraph._graph.node[hypothesesGraph._graph.node[n]['parent']]['trackId']
        if 'gap_parent' in hypothesesGraph._graph.node[n]:
            assert(trackId not in trackParents)
            gapTrackParents[trackId] = hypothesesGraph._graph.node[hypothesesGraph._graph.node[n]['gap_parent']]['trackId']

    # write res_track.txt
    getLogger().debug("Writing track text file")
    trackDict = {}
    for trackId, timestepList in tracks.items():
        timestepList.sort()
        if trackId in trackParents.keys():
            parent = trackParents[trackId]
        else:
            parent = 0
        # jumping over time frames, so creating 
        if trackId in gapTrackParents.keys():
            if gapTrackParents[trackId] != trackId:
                parent = gapTrackParents[trackId]
                getLogger().info("Jumping over one time frame in this link: trackid: {}, parent: {}, time: {}".format(trackId, parent, min(timestepList)))
        trackDict[trackId] = [parent, min(timestepList), max(timestepList)]
    save_tracks(trackDict, args) 

    # load images, relabel, and export relabeled result
    getLogger().debug("Saving relabeled images")
    pluginManager = TrackingPluginManager(verbose=args.verbose, pluginPaths=args.pluginPaths)
    pluginManager.setImageProvider('LocalImageLoader')
    imageProvider = pluginManager.getImageProvider()
    timeRange = imageProvider.getTimeRange(args.label_image_filename, args.label_image_path)

    for timeframe in range(timeRange[0], timeRange[1]):
        label_image = imageProvider.getLabelImageForFrame(args.label_image_filename, args.label_image_path, timeframe)

        # check if frame is empty
        if timeframe in mappings.keys():
            remapped_label_image = remap_label_image(label_image, mappings[timeframe])
            save_frame_to_tif(timeframe, remapped_label_image, args)
        else:
            save_frame_to_tif(timeframe, label_image, args)
