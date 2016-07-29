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

    assert(len(probabilityGenerator.TraxelsPerFrame[0]) == 4)
    assert(len(probabilityGenerator.TraxelsPerFrame[1]) == 3)
    assert(len(probabilityGenerator.TraxelsPerFrame[2]) == 3)
    assert(len(probabilityGenerator.TraxelsPerFrame[3]) == 4)
    filenamesPerTraxel = [t.segmentationFilename for t in probabilityGenerator.TraxelsPerFrame[3].values()]
    idsPerTraxel = [t.idInSegmentation for t in probabilityGenerator.TraxelsPerFrame[3].values()]
    assert(idsPerTraxel.count(1) == 2)
    assert(idsPerTraxel.count(2) == 2)
    assert(filenamesPerTraxel.count('tests/multiSegmentationHypothesesTestDataset/segmentation.h5') == 2)
    assert(filenamesPerTraxel.count('tests/multiSegmentationHypothesesTestDataset/segmentationAlt.h5') == 2)

if __name__ == "__main__":
    test_twoSegmentations()