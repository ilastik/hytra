from yapsy.IPlugin import IPlugin


class ObjectFeatureComputationPlugin(IPlugin):
    """
    This is the base class for all plugins that can do feature computation based on an image,
    which is a certain frame of the image sequence.
    They should implement the functions below.
    """

    #  specify names of features that are computed but should be ignored in the following
    omittedFeatures = []

    # specify for which dimensionality these features work
    worksForDimensions = [2, 3]

    def activate(self):
        """
        Activation of plugin could do something, but not needed here
        """
        pass

    def deactivate(self):
        """
        Deactivation of plugin could do something, but not needed here
        """
        pass

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        """
        Compute new features based on the raw and labelimage, for the given frame number.
        Should return a dict that has each computed feature as key, and as value a list
        of feature values for all objects in the frame (including object=0 which is assumed to be background).

        E.g.: Two features computed in a frame of 3 objects
        return {'meanIntensity': [0.1, 2.0, 2.7, 1.6],
                'center' : [np.array([100,100]), np.array([24,87]), np.array([65,12]), np.array([99,33])]}
        """
        raise NotImplementedError()

        return dict()
