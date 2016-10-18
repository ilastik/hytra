from yapsy.IPlugin import IPlugin


class MergerResolverPlugin(IPlugin):
    """
    This is the base class for all plugins that can resolve mergers in an image,
    using some initialization.
    They should implement the functions below.
    """

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

    def resolveMergerForCoords(self, coordinates, mergerCount, initializations=[]):
        """
        Resolve the pixel coordinates belonging to an object ID, into `mergerCount`
        new segments by fitting some kind of model. The `initializations` provide fits
        in the preceding frame of all possible incomings (list may be empty, but could
        also be more than `mergerCount`).
  
        `coordinates` pixel coordinates that belong to a merger ID in labelImage
        
        `mergerCount` number of gaussians to fit
  
        **returns** a list of fitted objects
        """
        raise NotImplementedError()

        return []

    def resolveMerger(self, labelImage, objectId, nextId, mergerCount, initializations=[]):
        """
        Resolve the object with the ID `objectId` in the `labelImage` into `mergerCount`
        new segments by fitting some kind of model. The `initializations` provide fits
        in the preceding frame of all possible incomings (list may be empty, but could
        also be more than `mergerCount`).

        `labelImage` is used read-only, use `updateLabelImage` to refine the segmentation

        **returns** a list of fitted objects
        """
        raise NotImplementedError()

        return []
    
    def updateLabelImage(self, labelImage, objectId, fits, newIds):
        """
        Resolve the object with the ID `objectId` in the `labelImage` into the fitted models with the given new IDs.
        `labelImage` should be updated by replacing all pixels that were labelled with `objectId`
        to get a new Id depending on the fit.
        """
        raise NotImplementedError()