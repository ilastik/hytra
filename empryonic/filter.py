from __future__ import print_function
from __future__ import unicode_literals
#
# (c) Bernhard X. Kausler, 2010
#

import h5py
import os, sys
import os.path as path
import numpy as np
import hdf5_format_spec
import vigra

fs = hdf5_format_spec.features
featurebasepath = fs['features_group']
labelcontent = fs['labelcontent_dataset']
labelcount = fs['supervoxels_dataset']
featureconfig = fs['featureconfig_dataset']
intminmax = 'intminmax'


def filterFeaturesByIntensity(h5In, h5Out, threshold = 1500):
    '''Remove label groups from the features group, with a maximal intensity below the threshold.
    
    h5In:  path to the hdf5 file to be filtered
    h5Out: output path; file will be overwritten if already existing
    '''
    def intensityFilter(labelGroup):
        intMaximum = labelGroup[intminmax].value[1]
        return (intMaximum >= threshold)
    filterFeaturesByPredicate(h5In, h5Out, intensityFilter)

def filterFeaturesByRandomForest(h5In, h5Out, rfFile, featsInds):
    '''
    Remove all features that are labeled as positive by a random forest classifier

    h5In: path to the HDF5 file to be filtered
    h5Out: output file name after filtering; will be overwritten if already existing
    rfFile: Name of the file where the RF classifier is stored
    featsInds: dictionary of features and indices that is required to construct the feature
               matrix
    '''
    rf = vigra.learning.RandomForest(rfFile,'RandomForest')
    nFeats = sum(len(featsInds[f]) for f in featsInds.keys())
    def rfFilter(labelGroup):
        idx = 0
        feats = np.zeros( (1, nFeats), dtype=np.float32 )
        for f in featsInds.keys():
            ds = labelGroup[f]
            feats[0,idx:idx+len(featsInds[f])] = ds[featsInds[f]] 
            idx += len(featsInds[f])
        lab = rf.predictLabels(feats)
        return lab[0]==0
    filterFeaturesByPredicate(h5In, h5Out, rfFilter)

def filterFeaturesByPredicate(h5In, h5Out, predicate):
    '''Remove label groups from the features group, that violate the predicate.

    predicate: predicate function, that takes a labelGroup as an argument
    h5In:  path to the hdf5 file to be filtered
    h5Out: output path; file will be overwritten if already existing
    '''
    
    if( h5In == h5Out ):
        raise ValueError("hdf5 in-place operation not supported (input and output path are identical)")

    inFile = h5py.File(h5In, 'r')
    
    #######################
    # filter by predicate #
    #######################
    featuresGroup = inFile[featurebasepath]
    labelGroups = filter(lambda item: isinstance(item, h5py.Group), featuresGroup.itervalues())
    
    validLabelGroups = filter( predicate, labelGroups )
    print("# of accepted cells: " + str(len(validLabelGroups)))

    #####################
    # write out results #
    #####################
    outFile = h5py.File(h5Out, 'w')
    outFeaturesGroup = outFile.create_group( featurebasepath )

    # supervoxels
    print("labelcount = ", labelcount)
    outFeaturesGroup.create_dataset(labelcount, data=featuresGroup[labelcount].value)

    # featureconfig
    outFeaturesGroup.create_dataset(featureconfig, data=featuresGroup[featureconfig].value)

    # labels
    for labelGroup in validLabelGroups:
        outFile.copy(labelGroup, outFeaturesGroup)

    # labelcontent
    inLabelcontent = featuresGroup[labelcontent].value
    outLabelcontent = np.zeros(inLabelcontent.shape, dtype=inLabelcontent.dtype)

    validLabels = filter(lambda item: item.isdigit(), outFeaturesGroup.keys())
    for label in validLabels:
        outLabelcontent[int(label) - 1] = 1
    outFeaturesGroup.create_dataset(labelcontent, data=np.array(outLabelcontent, dtype=np.uint16))

    inFile.close()
    outFile.close()
