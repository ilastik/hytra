import hytra.core.hypothesesgraph as hg
import networkx as nx

def test_trackletgraph():
    h = hg.HypothesesGraph()
    h._graph.add_path([0,1,2,3])
    for i in [0,1,2,3]:
        h._graph.node[i]['traxel'] = i
    
    t = h.generateTrackletGraph()
    assert(t.countArcs() == 0)
    assert(t.countNodes() == 1)

if __name__ == "__main__":
    test_trackletgraph()