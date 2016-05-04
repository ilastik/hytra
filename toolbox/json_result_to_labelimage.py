import configargparse as argparse
import logging
import h5py
import vigra
from vigra import numpy as np
import sys
import os
import json
sys.path.append('.')
from progressbar import ProgressBar

def get_uuid_to_traxel_map(traxelIdPerTimestepToUniqueIdMap):
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    uuidToTraxelMap = {}
    for t in timesteps:
        for i in traxelIdPerTimestepToUniqueIdMap[t].keys():
            uuid = traxelIdPerTimestepToUniqueIdMap[t][i]
            if uuid not in uuidToTraxelMap:
                uuidToTraxelMap[uuid] = []
            uuidToTraxelMap[uuid].append((int(t), int(i)))

    # sort the list of traxels per UUID by their timesteps
    for v in uuidToTraxelMap.values():
        v.sort(key=lambda timestepIdTuple: timestepIdTuple[0])

    return uuidToTraxelMap

def getLabelImageForFrame(labelImageFilename, labelImagePath, timeframe, shape):
    """
    Get the label image(volume) of one time frame
    """
    with h5py.File(labelImageFilename, 'r') as h5file:
        labelImage = h5file[labelImagePath % (timeframe, timeframe+1, shape[0], shape[1], shape[2])][0, ..., 0].squeeze().astype(np.uint32)
        return labelImage

def getShape(labelImageFilename, labelImagePath):
    """
    extract the shape from the labelimage
    """
    with h5py.File(labelImageFilename, 'r') as h5file:
        shape = h5file['/'.join(labelImagePath.split('/')[:-1])].values()[0].shape[1:4]
        return shape

def relabelImage(volume, replace):
    """
    Apply a set of label replacements to the given volume.

    Parameters:
        volume - numpy array
        replace - dictionary{[(oldValueInVolume)->(newValue), ...]}
    """
    mp = np.arange(0, np.amax(volume) + 1, dtype=volume.dtype)
    mp[1:] = 1
    labels = np.unique(volume)
    for label in labels:
        if label > 0:
            try:
                r = replace[label]
                mp[label] = r
            except:                
                pass
    return mp[volume]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform the segmentation as in ilastik for a new predicition map,'
                                                + 'using the same settings as stored in the given ilastik project',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--graph-json-file', required=True, type=str, dest='modelFilename',
                        help='Filename of the JSON graph model')
    parser.add_argument('--result-json-file', required=True, type=str, dest='resultFilename',
                        help='Filename of the JSON result file')
    parser.add_argument('--label-image-file', required=True, type=str, dest='labelImageFilename',
                        help='Filename of the HDF5/ILP file containing the segmentation')
    parser.add_argument('--label-image-path', type=str, dest='labelImagePath',
                        help='Path inside result file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument('--label-image-out', type=str, dest='out', required=True, help='Filename of the resulting HDF5 with relabeled objects')
    
    args, unknown = parser.parse_known_args()

    # get the dataset shape per frame:
    shape = getShape(args.labelImageFilename, args.labelImagePath)

    # load json model and results
    with open(args.modelFilename, 'r') as f:
        model = json.load(f)

    with open(args.resultFilename, 'r') as f:
        result = json.load(f)

    # load forward mapping and create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = model['traxelToUniqueId']
    uuidToTraxelMap = get_uuid_to_traxel_map(traxelIdPerTimestepToUniqueIdMap)

    # load links and map indices
    links = [(uuidToTraxelMap[int(entry['src'])][-1], uuidToTraxelMap[int(entry['dest'])][0]) for entry in result['linkingResults'] if entry['value'] > 0]

    # add all internal links of tracklets
    for v in uuidToTraxelMap.values():
        prev = None
        for timestepIdTuple in v:
            if prev is not None:
                links.append((prev, timestepIdTuple))
            prev = timestepIdTuple

    # group by timestep
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    linksPerTimestep = dict([(t, [(a[1], b[1]) for a, b in links if b[0] == int(t)]) for t in timesteps])
    assert(len(linksPerTimestep['0']) == 0)

    # create output array
    resultVolume = np.zeros((len(timesteps),) + shape, dtype='uint32')
    print("resulting volume shape: {}".format(resultVolume.shape))
    progressBar = ProgressBar(stop=len(timesteps))
    progressBar.show(0)

    # iterate over timesteps and label tracks from front to back in a distinct color
    nextUnusedColor = 1
    lastFrameColorMap = {}
    lastFrameLabelImage = getLabelImageForFrame(args.labelImageFilename, args.labelImagePath, 0, shape)
    for t in range(1,len(timesteps)):
        progressBar.show()
        thisFrameColorMap = {}
        thisFrameLabelImage = getLabelImageForFrame(args.labelImageFilename, args.labelImagePath, t, shape)
        for a, b in linksPerTimestep[str(t)]:
            # propagate color if possible, otherwise assign a new one
            if a in lastFrameColorMap:
                thisFrameColorMap[b] = lastFrameColorMap[a]
            else:
                thisFrameColorMap[b] = nextUnusedColor
                lastFrameColorMap[a] = thisFrameColorMap[b]  # also store in last frame's color map as it must have been present to participate in a link
                nextUnusedColor += 1

        # see which objects have been assigned a color in the last frame. set all others to 0 (1?)
        unusedLabels = set(np.unique(lastFrameLabelImage)) - set([0]) - set(lastFrameColorMap.keys())
        for l in unusedLabels:
            lastFrameColorMap[l] = 0

        # write relabeled image
        resultVolume[t-1,...,0] = relabelImage(lastFrameLabelImage, lastFrameColorMap)

        # swap the color maps so that in the next frame we use "this" as "last"
        lastFrameColorMap, thisFrameColorMap = thisFrameColorMap, lastFrameColorMap
        lastFrameLabelImage = thisFrameLabelImage

    # handle last frame:
    # see which objects have been assigned a color in the last frame. set all others to 0 (1?)
    unusedLabels = set(np.unique(lastFrameLabelImage)) - set([0]) - set(lastFrameColorMap.keys())
    for l in unusedLabels:
        lastFrameColorMap[l] = 0

    # write last frame relabeled image
    resultVolume[t,...,0] = relabelImage(lastFrameLabelImage, lastFrameColorMap)
    progressBar.show()

    # save to disk
    if os.path.exists(args.out):
        os.remove(args.out)
    vigra.impex.writeHDF5(resultVolume, args.out, 'exported_data')

