import networkx as nx
from sklearn.neighbors import KDTree

class HypothesesGraph:
	"""
	Replacement for pgmlink's hypotheses graph, 
	with a similar API so it can be used as drop-in replacement
	"""
	def __init__(self):
		self._graph = nx.DiGraph()

	def nodeIterator(self):
		return self._graph.nodes_iter()

	def arcIterator(self):
		return self._graph.edges_iter()

	def source(self, edge):
		return edge[0]

	def target(self, edge):
		return edge[1]

	def _findNearestNeighbors(self, kdtreeObjectPair, traxel, numNeighbors, maxNeighborDist):
		"""
		Return a list of object IDs which are the 'numNeighbors' closest elements 
		in the kdtree less than maxNeighborDist away of the traxel.
		"""
		kdtree, objectIdList = kdtreeObjectPair
		neighbors = kdtree.query(self._extractCenter(traxel), k=numNeighbors, return_distance=True)
		return [objectIdList[index] for distance, index in neighbors if distance < maxNeighborDist]

	def __getTraxelFeatureVector(self, traxel, featureName, maxNumDimensions=3):
	    result = []
	    for i in range(maxNumDimensions):
	        try:
	            result.append(traxel.get_feature_value(feature_name, i))
	        except:
	        	if i == 0:
	        		print("Error when accessing feature {}[{}] for traxel (Id={},Timestep={})".format(feature_name,
	                                                                                              i,
	                                                                                              traxel.Id,
	                                                                                              traxel.Timestep))
		            print "Available features are: "
		            print traxel.print_available_features()
		            raise Exception
		        else:
	            	pass
	    return result

	def _extractCenter(self, traxel):
		try:
			# python traxelstore
			if 'com' in traxel.Features:
				return traxel.Features['com']
			else:
				return traxel.Features['RegionCenter']
		except:
			# C++ pgmlink traxelstore
			try:
				return self.__getTraxelFeatureVector(traxel, 'com')
			except:
				try:
					return self.__getTraxelFeatureVector(traxel, 'RegionCenter')
				except:
					raise InvalidArgumentException('given traxel (t={},id={}) does not have \
						"com" or "RegionCenter"'.format(traxel.Timestep, traxel.Id))


	def _traxelMightDivide(self, traxel, divisionThreshold):
		assert 'divProb' in traxel.Features
		return traxel.Features['divProb'][0] > divisionThreshold

	def _buildFrameKdTree(self, traxelDict):
		"""
		Collect the centers of all traxels and their ids of this frame's traxels.
		Then build a kdtree and return (kdtree, listOfObjectIdsInFrame), where the second argument
		is needed to decode the object id of the nearest neighbors in _findNearestNeighbors().
		"""
		objectIdList = []
		features = []
		for obj, traxel in traxelDict.iteritems():
			if obj == 0:
				continue
			objectIdList.append(obj)
			features.append(list(self._extractCenter(traxel)))

		return (KDTree(features, metric='euclidean'), objectIdList)

	def _addNodesForFrame(self, frame, traxelDict):
		"""
		Insert nodes for all objects in this frame, with the attribute "traxel"
		"""
		for obj, traxel in traxelDict.iteritems():
			if obj == 0:
				continue
			self._graph.add_node((obj, frame), traxel=traxel)

	def buildFromTraxelstore(self, traxelstore, maxNeighborDist=200, numNearestNeighbors=1, 
							 forwardBackwardCheck=True, divisionThreshold=0.1):
		"""
		Takes a python traxelstore containing traxel features and finds probable links between frames.
		"""
		assert(traxelstore is not None and len(traxelstore.TraxelsPerFrame) > 0)

		for frame in range(len(traxelstore.TraxelsPerFrame.keys()) - 1):
			if frame > 0:
				kdTreeThisFrame = kdTreeNextFrame
			else:
				kdTreeThisFrame = self._buildFrameKdTree(traxelstore.TraxelsPerFrame[frame])
				self._addNodesForFrame(frame, traxelstore.TraxelsPerFrame[frame])

			kdTreeNextFrame = self._buildFrameKdTree(traxelstore.TraxelsPerFrame[frame + 1])
			self._addNodesForFrame(frame, traxelstore.TraxelsPerFrame[frame+1])
			
			# find forward links
			for obj, traxel in traxelstore.TraxelsPerFrame[frame].iteritems():
				divisionPreservingNumNearestNeighbors = numNearestNeighbors
				if divisionPreservingNumNearestNeighbors < 2 and self._traxelMightDivide(traxel, divisionThreshold):
					divisionPreservingNumNearestNeighbors = 2
				neighbors = self._findNearestNeighbors(knTreeNextFrame, traxel, numNearestNeighbors)
				for n in neighbors:
					self._graph.add_edge((frame, obj), (frame+1, n))

			# find backward links
			if forwardBackwardCheck:
				for obj, traxel in traxelstore.TraxelsPerFrame[frame+1].iteritems():
					neighbors = self._findNearestNeighbors(knTreeThisFrame, traxel, numNearestNeighbors)
				for n in neighbors:
					self._graph.add_edge((frame, n), (frame+1, obj))

	def generateTrackletGraph():
		raise NotImplementedError()

