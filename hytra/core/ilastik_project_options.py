import h5py

class IlastikProjectOptions:
    """
    The Ilastik Project Options configure where in the project HDF5 file the important things can be found.
    Use this when creating a Traxelstore
    """

    def __init__(self):
        self.objectCountClassifierFile = None
        self.objectCountClassifierPath = '/CountClassification'
        self.divisionClassifierFile = None
        self.divisionClassifierPath = '/DivisionDetection'
        self.transitionClassifierFile = None
        self.transitionClassifierPath = None
        self.selectedFeaturesGroupName = 'SelectedFeatures'
        self.classifierForestsGroupName = 'ClassifierForests'
        self.randomForestZeroPaddingWidth = 4
        self.labelImageFilename = None
        self.labelImagePath = '/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'
        self.rawImageFilename = None
        self.rawImagePath = None
        self.rawImageAxes = None
        self.imageProviderName = 'LocalImageLoader'
        self.featureSerializerName = 'LocalFeatureSerializer'
        self.sizeFilter = None  # set to tuple with min,max pixel count

def extractWeightDictFromIlastikProject(ilpFilename, basePath='/ConservationTracking/Parameters/0000'):
    """
    Open an ilastik tracking project and extract the conservation tracking parameters
    that weigh the contribution of the different energies/classifiers.

    **Parameters:**
    * `ilpFilename`: filename of the ilastik conservation tracking project

    **Returns** a dictionary with weights that can be passed on to the solvers directly
    """
    with h5py.File(ilpFilename, 'r') as h5file:
        withDivisions = bool(h5file[basePath + '/withDivisions'].value)
        transitionWeight = float(h5file[basePath + '/transWeight'].value)
        detectionWeight = 10.0
        divisionWeight = float(h5file[basePath + '/divWeight'].value)
        appearanceWeight = float(h5file[basePath + '/appearanceCost'].value)
        disappearanceWeight = float(h5file[basePath + '/disappearanceCost'].value)
    
    if withDivisions:
        weights = {'weights' : [transitionWeight, detectionWeight, divisionWeight, appearanceWeight, disappearanceWeight]}
    else:
        weights = {'weights' : [transitionWeight, detectionWeight, appearanceWeight, disappearanceWeight]}

    return weights
