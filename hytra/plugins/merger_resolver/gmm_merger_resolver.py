from hytra.pluginsystem import merger_resolver_plugin
import numpy as np

from sklearn import mixture


class GMMMergerResolver(merger_resolver_plugin.MergerResolverPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    def initGMM(self, mergerCount, object_init_list):
        gmm = mixture.GMM(n_components=mergerCount)
        if len(object_init_list) > 0:
            gmm.weights_ = np.array([o[0] for o in object_init_list])
            gmm.covars_ = np.array([o[1] for o in object_init_list])
            gmm.means_ = np.array([o[2] for o in object_init_list])
        return gmm

    def getObjectInitializationList(self, gmm):
        return zip(gmm.weights_, gmm.covars_, gmm.means_)

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
  
        # fit GMM to label image data
        gmm = self.initGMM(mergerCount, initializations)
        gmm.fit(coordinates)
        assert(gmm.converged_)
  
        return self.getObjectInitializationList(gmm)


    def resolveMerger(self, labelImage, objectId, nextId, mergerCount, initializations=[]):
        """
        Resolve the object with the ID `objectId` in the `labelImage` into `mergerCount`
        new segments by fitting some kind of model. The `initializations` provide fits
        in the preceding frame of all possible incomings (list may be empty, but could
        also be more than `mergerCount`).
  
        `labelImage` is used read-only, use `updateLabelImage` to refine the segmentation
  
        **returns** a list of fitted objects
        """
  
        # fit GMM to label image data
        coordinates = np.transpose(np.vstack(np.where(labelImage == objectId)))
        gmm = self.initGMM(mergerCount, initializations)
        gmm.fit(coordinates)
        assert(gmm.converged_)
  
        return self.getObjectInitializationList(gmm)

    def updateLabelImage(self, labelImage, objectId, fits, newIds, offset=[0,0,0]):
        """
        Resolve the object with the ID `objectId` in the `labelImage` into the fitted models with the given new IDs.
        `labelImage` should be updated by replacing all pixels that were labelled with `objectId`
        to get a new Id depending on the fit.
        """
        
        if len(fits) > 1:
            assert(len(fits) == len(newIds))
            # edit labelimage in-place
            coordinates = np.transpose(np.vstack(np.where(labelImage == objectId)))
            coordinates = coordinates + offset
            gmm = self.initGMM(len(fits), fits)
            responsibilities = gmm.predict(coordinates)
            newIds = np.array(newIds)
            newObjectIds = newIds[responsibilities]
            labelImage[labelImage == objectId] = newObjectIds