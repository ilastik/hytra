#
# (c) Bernhard X. Kausler, 2010
#

import os.path
import sys

import vigra

# try to import python wrapper from project source
module_path_in_source = 'python_wrapper/'
if(os.path.exists(module_path_in_source) or os.path.exists('../' + module_path_in_source)):
   sys.path.append(module_path_in_source)
   sys.path.append('../' + module_path_in_source)
import segmentation as seg



def msa( volume,
         scales = [1.6, 2.4],
         opening_radius = 1,
         closing_radius = 2,
         thresholds = [-1, -1.5, -2.0]):
    '''Segment the volumetric data with the MSA algorithm.

    Internally, we use numpy.float32 precision.
    
    :param volume: a three-dimensional numpy.ndarray of scalars
    '''
    s = seg.FloatVec()
    s.extend( scales )
    t = seg.FloatVec()
    t.extend( thresholds )

    segmenter = seg.ctSegmentationMSA(s, opening_radius, closing_radius, t)

    # vigranumpy transposes the indices;
    # that is not desirable for our use, so we transpose them back
    in_data = vigra.ScalarVolume(volume.transpose())
    return segmenter.run(in_data)

