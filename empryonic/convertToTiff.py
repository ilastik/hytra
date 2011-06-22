#!/usr/bin/python

import h5py
import vigra
import os
import numpy
import math
def convertToTiff(h5filename, stackdir, dataTransposed=True, threshold=100,
                  scaleFactor=16):
    """
    Read a raw 3D volume from the file h5filename, cut it into slices,
    convert it to 8bit and save it as a stack of TIFF images.
    h5filename : name of HDF5 file, from which the raw volume is read
    stackdir : directory where the TIFF stack shall be created
    dataTransposed : if 'True', the data has been transposed (nZ,nY,nX)
                     if 'False', it is in the correct order (nX,nY,nZ)
    threshold : all pixel values below the threshold are truncated to 0
    scaleFactor : factor by which the data is scaled (after the threshold has been
                  subtracted)
    """
    fid = h5py.File(h5filename, 'r')
    data = fid['/raw/volume']
    stackdir = stackdir + '/'
    if dataTransposed:
        nSlices = data.shape[0]
    else:
        nSlices = data.shape[2]
    # The following number is the width of the number field, by which the
    # image slices are indexed. In theory, it should be floor(...)+1
    # instead of ceil(...)+1, but we err on the safe side due to possible
    # rounding errors.
    fieldWidth=math.ceil(math.log(nSlices)/math.log(10))+1 
    fileName = 'slice_%%0%du.tiff' % fieldWidth
    if not os.access(stackdir, os.F_OK):
        os.mkdir(stackdir)
    for i in range(nSlices):
        if dataTransposed:
            imSlice = numpy.transpose(data[i,:,:])
        else:
            imSlice = data[:,:,i]
        imSlice[imSlice<threshold]=threshold
        imSlice = (imSlice-threshold)/scaleFactor
        imSlice[imSlice>255]=255
        vigra.impex.writeImage(imSlice, stackdir+(fileName % i), \
                               dtype='UINT8')
    fid.close()

