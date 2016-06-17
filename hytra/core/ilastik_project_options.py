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