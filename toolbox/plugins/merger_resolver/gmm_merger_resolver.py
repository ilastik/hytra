from toolbox.pluginsystem import merger_resolver_plugin
import numpy as np

from sklearn import mixture


class GMMMergerResolver(merger_resolver_plugin.MergerResolverPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    def initGMM(self,mergerCount,object_init_list):
        gmm = mixture.GMM(n_components=mergerCount)
        if len(object_init_list) > 0:
            gmm.weights_ = np.array([o[0] for o in object_init_list])
            gmm.covars_ = np.array([o[1] for o in object_init_list])
            gmm.means_ = np.array([o[2] for o in object_init_list])
        return gmm

    def getObjectInitializationList(self,gmm):
        return zip(gmm.weights_,gmm.covars_,gmm.means_)

    def resolveMerger(self, labelImage, objectId, nextId, mergerCount, initializations=[]):
        """
        Resolve the object with the ID `objectId` in the `labelImage` into `mergerCount`
        new segments by fitting a Gaussian Mixture Model. The `initializations` provide fits
        in the preceding frame of all possible incomings (list may be empty, but could
        also be more than `mergerCount`).

        `labelImage` should be updated by replacing all pixels that were labelled with `objectId`
        to get a new Id depending on the fit, starting from `nextId`.

        **returns** a list of fitted objects
        """

        # fit GMM to label image data
        coordinates =  np.transpose(np.vstack(np.where(labelImage==objectId)))
        gmm = self.initGMM(mergerCount,initializations)
        gmm.fit(coordinates)

        # edit labelimage in-place
        responsibilities = gmm.predict(coordinates)
        newObjectIds = responsibilities + nextId
        labelImage[labelImage==objectId] = newObjectIds

        return self.getObjectInitializationList(gmm)

