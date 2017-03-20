from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import logging

from hytra.core.ilastik_project_options import IlastikProjectOptions
from hytra.jst.conflictingsegmentsprobabilitygenerator import ConflictingSegmentsProbabilityGenerator
from hytra.core.ilastikhypothesesgraph import IlastikHypothesesGraph
from hytra.core.fieldofview import FieldOfView

try:
    import multiHypoTracking_with_cplex as mht
except ImportError:
    try:
        import multiHypoTracking_with_gurobi as mht
    except ImportError:
        mht = None
        import dpct


def constructFov(shape, t0, t1, scale=[1, 1, 1]):
    [xshape, yshape, zshape] = shape
    [xscale, yscale, zscale] = scale

    fov = FieldOfView(t0, 0, 0, 0, t1, xscale * (xshape - 1), yscale * (yshape - 1),
                      zscale * (zshape - 1))
    return fov

# def test_twoSegmentations():
#     # set up ConflictingSegmentsProbabilityGenerator
#     ilpOptions = IlastikProjectOptions()
#     ilpOptions.divisionClassifierPath = None
#     ilpOptions.divisionClassifierFilename = None
    
#     ilpOptions.rawImageFilename = 'tests/multiSegmentationHypothesesTestDataset/Raw.h5'
#     ilpOptions.rawImagePath = 'exported_data'
#     ilpOptions.rawImageAxes = 'txyzc'

#     ilpOptions.labelImageFilename = 'tests/multiSegmentationHypothesesTestDataset/segmentation.h5'

#     ilpOptions.objectCountClassifierFilename = 'tests/multiSegmentationHypothesesTestDataset/tracking.ilp'

#     additionalLabelImageFilenames = ['tests/multiSegmentationHypothesesTestDataset/segmentationAlt.h5']
#     additionalLabelImagePaths = [ilpOptions.labelImagePath]

#     probabilityGenerator = ConflictingSegmentsProbabilityGenerator(
#         ilpOptions, 
#         additionalLabelImageFilenames,
#         additionalLabelImagePaths,
#         useMultiprocessing=False,
#         verbose=False)
#     probabilityGenerator.fillTraxels(usePgmlink=False)

#     assert(len(probabilityGenerator.TraxelsPerFrame[0]) == 4)
#     assert(len(probabilityGenerator.TraxelsPerFrame[1]) == 3)
#     assert(len(probabilityGenerator.TraxelsPerFrame[2]) == 3)
#     assert(len(probabilityGenerator.TraxelsPerFrame[3]) == 4)
#     filenamesPerTraxel = [t.segmentationFilename for t in probabilityGenerator.TraxelsPerFrame[3].values()]
#     idsPerTraxel = [t.idInSegmentation for t in probabilityGenerator.TraxelsPerFrame[3].values()]
#     assert(idsPerTraxel.count(1) == 2)
#     assert(idsPerTraxel.count(2) == 2)
#     assert(filenamesPerTraxel.count('tests/multiSegmentationHypothesesTestDataset/segmentation.h5') == 2)
#     assert(filenamesPerTraxel.count('tests/multiSegmentationHypothesesTestDataset/segmentationAlt.h5') == 2)

#     # build hypotheses graph, check that conflicting traxels are properly detected
#     fieldOfView = constructFov(probabilityGenerator.shape,
#                                probabilityGenerator.timeRange[0],
#                                probabilityGenerator.timeRange[1],
#                                [probabilityGenerator.x_scale,
#                                 probabilityGenerator.y_scale,
#                                 probabilityGenerator.z_scale])
 
#     hypotheses_graph = IlastikHypothesesGraph(
#         probabilityGenerator=probabilityGenerator,
#         timeRange=probabilityGenerator.timeRange,
#         maxNumObjects=1,
#         numNearestNeighbors=2,
#         fieldOfView=fieldOfView,
#         withDivisions=False,
#         divisionThreshold=0.1
#     )

#     assert(hypotheses_graph.countNodes() == 14)
#     assert(hypotheses_graph.countArcs() == 23)
#     assert(hypotheses_graph._graph.node[(0, 1)]['traxel'].conflictingTraxelIds == [3])
#     assert(hypotheses_graph._graph.node[(0, 3)]['traxel'].conflictingTraxelIds == [1])
#     assert(hypotheses_graph._graph.node[(0, 2)]['traxel'].conflictingTraxelIds == [4])
#     assert(hypotheses_graph._graph.node[(0, 4)]['traxel'].conflictingTraxelIds == [2])
#     assert(hypotheses_graph._graph.node[(1, 1)]['traxel'].conflictingTraxelIds == [2, 3])
#     assert(hypotheses_graph._graph.node[(1, 2)]['traxel'].conflictingTraxelIds == [1])
#     assert(hypotheses_graph._graph.node[(1, 3)]['traxel'].conflictingTraxelIds == [1])

#     # track, but check that the right exclusion constraints are present
#     hypotheses_graph.insertEnergies()
#     trackingGraph = hypotheses_graph.toTrackingGraph()

#     assert(len(trackingGraph.model['exclusions']) == 8)
#     for exclusionSet in trackingGraph.model['exclusions']:
#         assert(len(exclusionSet) == 2)

#     # use multiHypoTracking, insert exclusion constraints!
#     if mht is not None:
#         result = mht.track(trackingGraph.model, {"weights": [10, 10, 500, 500]})
#     else:
#         return
#         # standard dpct cannot handle exclusion constraints yet
#         result = dpct.trackFlowBased(trackingGraph.model, {"weights": [10, 10, 500, 500]})

#     hypotheses_graph.insertSolution(result)
#     # hypotheses_graph.computeLineage()

#     numActivePerFrame = {}

#     for node in hypotheses_graph.nodeIterator():
#         timeframe = node[0]
#         if 'value' in hypotheses_graph._graph.node[node]: 
#             value = hypotheses_graph._graph.node[node]['value']
#         else:
#             value = 0 
#         numActivePerFrame.setdefault(timeframe, []).append(value) 

#     for _, v in numActivePerFrame.iteritems():
#         assert(sum(v) == 2)        

#     edgeFlow = 0
#     for edge in hypotheses_graph.arcIterator():
#         if 'value' in hypotheses_graph._graph.edge[edge[0]][edge[1]]:
#             edgeFlow += hypotheses_graph._graph.edge[edge[0]][edge[1]]['value']
#     assert(edgeFlow == 6)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_twoSegmentations()
