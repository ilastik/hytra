# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import argparse
import h5py
import vigra
from vigra import numpy as np
import hytra.util.axesconversion
from hytra.util.progressbar import ProgressBar

def filter_labels(a, min_size, max_size=None):
    """
    Remove (set to 0) labeled connected components that are too small or too large.
    Note: Operates in-place.
    """
    if min_size == 0 and (max_size is None or max_size > np.prod(a.shape)): # shortcut for efficiency
        return a

    try:
        component_sizes = np.bincount( a.ravel() )
    except TypeError:
        # On 32-bit systems, must explicitly convert from uint32 to int
        # (This fix is just for VM testing.)
        component_sizes = np.bincount( np.asarray(a.ravel(), dtype=int) )

    bad_sizes = component_sizes < min_size
    if max_size is not None:
        np.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )
    
    bad_locations = bad_sizes[a]
    a[bad_locations] = 0
    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform the segmentation as in ilastik for a new predicition map,'
                                                + 'using the same settings as stored in the given ilastik project',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ilastik-project', required=True, type=str, dest='ilpFilename',
                        help='Filename of the ilastik project')
    parser.add_argument('--prediction-map', required=True, type=str, dest='predictionMapFilename',
                        help='Filename of the hdf5 file containing the prediction maps output by ilastik')
    parser.add_argument('--prediction-path', type=str, dest='predictionPath', default='exported_data',
                        help='Path inside HDF5 file to prediction map')
    parser.add_argument('--label-image-path', type=str, dest='labelImagePath',
                        help='Path inside result file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument("--prediction-axes", dest='prediction_axes', required=True, type=str,
                        help="axes ordering of the prediction map, e.g. txyzc")
    parser.add_argument('--out', type=str, dest='out', required=True, help='Filename of the resulting HDF5 labelimage')
    
    args = parser.parse_args()
    
    # load threshold settings
    with h5py.File(args.ilpFilename, 'r') as h5file:
        threshold_level = h5file['ThresholdTwoLevels/SingleThreshold'].value
        threshold_channel = h5file['ThresholdTwoLevels/Channel'].value
        threshold_sigmas = [h5file['ThresholdTwoLevels/SmootherSigma/x'].value, 
                            h5file['ThresholdTwoLevels/SmootherSigma/y'].value, 
                            h5file['ThresholdTwoLevels/SmootherSigma/z'].value]
        threshold_min_size = h5file['ThresholdTwoLevels/MinSize'].value
        threshold_max_size = h5file['ThresholdTwoLevels/MaxSize'].value

    # load prediction maps
    # predictionMaps = vigra.impex.readHDF5(args.predictionMapFilename, args.predictionPath)
    with h5py.File(args.predictionMapFilename) as f:
        predictionMaps = f[args.predictionPath].value

    ndim = len(predictionMaps.squeeze().shape) - 2
    predictionMaps = hytra.util.axesconversion.adjustOrder(predictionMaps, args.prediction_axes, 'txyzc')
    shape = predictionMaps.shape

    print("Found PredictionMaps of shape {} (txyzc), using channel {}".format(
        predictionMaps.shape, 
        threshold_channel))

    progressBar = ProgressBar(stop=shape[0])
    progressBar.show(0)

    with h5py.File(args.out, 'w') as h5file:
        # loop over timesteps
        for t in range(shape[0]):
            # smooth, threshold, and size filter from prediction map
            framePrediction = predictionMaps[t, ..., threshold_channel].squeeze()
            assert ndim == len(framePrediction.shape)

            smoothedFramePrediction = vigra.filters.gaussianSmoothing(framePrediction.astype('float32'), threshold_sigmas[:ndim])
            foreground = np.zeros(smoothedFramePrediction.shape, dtype='uint32')
            foreground[smoothedFramePrediction > threshold_level] = 1

            if ndim == 2:
                labelImage = vigra.analysis.labelImageWithBackground(foreground)
            else:
                labelImage = vigra.analysis.labelVolumeWithBackground(foreground)

            # filter too small / too large objects out
            filter_labels(labelImage, threshold_min_size, threshold_max_size)

            # run labelImage again to get consecutive IDs
            if ndim == 2:
                labelImage = vigra.analysis.labelImageWithBackground(labelImage)
            else:
                labelImage = vigra.analysis.labelVolumeWithBackground(labelImage)

            # bring to right size
            if ndim == 2:
                axes = 'xy'
            else:
                axes = 'xyz'

            labelImage = hytra.util.axesconversion.adjustOrder(labelImage, axes, 'txyzc')

            # save
            h5file[args.labelImagePath % (t, t+1, shape[1], shape[2], shape[3])] = labelImage
            progressBar.show()


