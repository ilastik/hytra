import vigra

from ..types.object_feature_computation import ObjectFeatureComputationBase


class StandardObjectFeatures(ObjectFeatureComputationBase):
    """
    Computes the standard vigra region features
    """

    worksForDimensions = [2, 3]
    omittedFeatures = [
        "Global<Maximum >",
        "Global<Minimum >",
        "Histogram",
        "Weighted<RegionCenter>",
    ]

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        return vigra.analysis.extractRegionFeatures(
            rawImage.squeeze().astype("float32"),
            labelImage.squeeze().astype("uint32"),
            ignoreLabel=0,
        )
