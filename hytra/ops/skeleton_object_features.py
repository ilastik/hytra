import vigra
from vigra import numpy as np

from ..types.object_feature_computation import ObjectFeatureComputationBase


class SkeletonObjectFeatures(ObjectFeatureComputationBase):
    """
    Computes the skeleton based vigra features
    """

    worksForDimensions = [2]
    omittedFeatures = ["Polygon"]

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        featureDict = vigra.analysis.extractSkeletonFeatures(labelImage.squeeze().astype(np.uint32))
        if "Center" in featureDict:
            # old vigra versions simply call that feature "Center" which conflicts with other features
            featureDict["Skeleton Center"] = featureDict["Center"]
            del featureDict["Center"]
        return featureDict
