import logging

from hytra.core.ilastik_project_options import IlastikProjectOptions
from hytra.core.conflictingsegmentsprobabilitygenerator import ConflictingSegmentsProbabilityGenerator

def test_twoSegmentations():
    logging.basicConfig(level=logging.INFO)

    ilpOptions = IlastikProjectOptions()
    ilpOptions.divisionClassifierPath = None
    ilpOptions.divisionClassifierFilename = None
    
    ilpOptions.rawImageFilename = 'tests/multiSegmentationHypothesesTestDataset/Raw.h5'
    ilpOptions.rawImagePath = 'exported_data'
    ilpOptions.rawImageAxes = 'txyzc'

    ilpOptions.labelImageFilename = 'tests/multiSegmentationHypothesesTestDataset/segmentation.h5'

    ilpOptions.objectCountClassifierFilename = 'tests/multiSegmentationHypothesesTestDataset/tracking.ilp'

    additionalLabelImageFilenames = ['tests/multiSegmentationHypothesesTestDataset/segmentationAlt.h5']
    additionalLabelImagePaths = [ilpOptions.labelImagePath]

    probabilityGenerator = ConflictingSegmentsProbabilityGenerator(
        ilpOptions, 
        additionalLabelImageFilenames,
        additionalLabelImagePaths,
        useMultiprocessing=False,
        verbose=False)
    probabilityGenerator.fillTraxels(usePgmlink=False)