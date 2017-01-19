## Copyright (c) 2010 Bernhard X. Kausler

## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

'''
Associate elements of two arbitrarily sized sets such that total association
costs are minimzed. Nonmatches are allowed and controlled by a separate cost function.
'''
from __future__ import unicode_literals

from builtins import map
from builtins import str
from builtins import object
import collections as _collections
import pulp as _pulp



class _BipartiteGraph( object ):
    '''Data structure to store an undirected bipartite graph.'''
    
    Vertex = _collections.namedtuple('Vertex', 'id value edges')
    Edge = _collections.namedtuple('Edge', 'id id_lhs id_rhs weight')
    def __init__( self ):
        self._lhs = dict()
        self._rhs = dict()
        self._edges = dict()

    @property
    def lhs( self ):
        return self._lhs

    @property
    def rhs( self ):
        return self._rhs

    @property
    def edges( self ):
        return self._edges
    
    def add_vertex( self, id, value, lhs_or_rhs):
        v = self.Vertex(id, value, [])
        if lhs_or_rhs == 'lhs':
            self._lhs[id] = v
        elif lhs_or_rhs == 'rhs':
            self._rhs[id] = v
        else:
            raise Exception("use one of the following as partition identifier: 'lhs' or 'rhs'")

    def add_edge( self, id, id_lhs, id_rhs, weight ):
        if not(id_rhs in self.rhs and id_lhs in self.lhs):
            raise Exception('cannot add edge between non-existing vertices')
        
        edge = self.Edge(id, id_lhs, id_rhs, weight )
        self.edges[id] = edge
        self.lhs[id_lhs].edges.append(edge)
        self.rhs[id_rhs].edges.append(edge)



def _construct_match_graph(candidates_lhs, candidates_rhs, match_cost_function, nonmatch_cost_function, cost_threshold=None):
    '''Encode matching problem as a bipartite graph.

    A bipartite graph is construced from the two candidate sets. Additionally, a sink vertex is added to
    both partition of the graph to represent nonmatching. Matching costs are encoded as weights.

    cost_threshold -- only add edges, when their weight is below this threshold; applies only to match costs
    '''
    graph = _BipartiteGraph()
    [graph.add_vertex(x[0], x[1], 'lhs') for x in enumerate(candidates_lhs)]
    [graph.add_vertex(x[0], x[1], 'rhs') for x in enumerate(candidates_rhs)]
    # list(map(lambda x: graph.add_vertex(x[0], x[1], 'lhs'), enumerate(candidates_lhs)))
    # list(map(lambda x: graph.add_vertex(x[0], x[1], 'rhs'), enumerate(candidates_rhs)))

    # connect vertices (if not above cost threshold)
    for vertex_lhs in list(graph.lhs.values()):
        for vertex_rhs in list(graph.rhs.values()):
            cost = match_cost_function(vertex_lhs.value, vertex_rhs.value)
            if (not cost_threshold) or (cost_threshold and cost < cost_threshold):
                graph.add_edge('edge_' + str(vertex_lhs.id)+'_'+str(vertex_rhs.id), vertex_lhs.id, vertex_rhs.id, cost)
                
    # add sink nodes to represent nonmatches
    graph.add_vertex('sink', None, 'lhs')
    graph.add_vertex('sink', None, 'rhs')
    for vertex in list(graph.lhs.values()):
        if vertex.id != 'sink':
            graph.add_edge('edge_'+str(vertex.id)+'_sink', vertex.id, 'sink', nonmatch_cost_function(vertex.value))
    for vertex in list(graph.rhs.values()):
        if vertex.id != 'sink':
            graph.add_edge('edge_sink_'+str(vertex.id), 'sink', vertex.id, nonmatch_cost_function(vertex.value))

    return graph


def _formulate_integer_linear_program( graph ):
    '''Formulate a matching problem encoded in a bipartite graph as an ilp.

    Returns a 'pulp' integer linear problem. Solve it by calling the method solve(). 
    graph -- instance of _BipartiteGraph

    '''
    ilp = _pulp.LpProblem('bipartite_matching', _pulp.LpMinimize)

    # represent every edge in the bipartite graph with a binary variable
    match_vars = dict()
    for edge_id in list(graph.edges.keys()):
        match_vars[edge_id] = _pulp.LpVariable(edge_id, 0, 1, _pulp.LpInteger)
    ilp += _pulp.lpSum([graph.edges[id].weight * var for id,var in list(match_vars.items())])

    # allow max. one edge per vertex, except the sink vertices
    for vertex in list(graph.lhs.values()):
        if vertex.id != 'sink':
            ilp += _pulp.lpSum([match_vars[edge.id] for edge in vertex.edges]) == 1, ''
    for vertex in list(graph.rhs.values()):
        if vertex.id != 'sink':
            ilp += _pulp.lpSum([match_vars[edge.id] for edge in vertex.edges]) == 1, ''

    return ilp, match_vars


def _formulate_associations( graph, solved_ilp_variables ):
    assoc = dict()
    assoc['lhs'] = dict()
    assoc['rhs'] = dict()
    for id, var in list(solved_ilp_variables.items()):
        if var.value() == 1:
            match = graph.edges[id]
            lhs = graph.lhs[match.id_lhs].value
            rhs = graph.rhs[match.id_rhs].value
            if lhs != None:
                assoc['lhs'][lhs] = rhs  
            if rhs != None:
                assoc['rhs'][rhs] = lhs
    return assoc



def optimal_matching(candidates_lhs, candidates_rhs, match_cost_function, nonmatch_cost_function, cost_threshold = None):
    '''Match two sets of candidates by minimizing total association costs.

    The elements at the left-hand side (lhs) are paired with the elements at the right-hand side (rhs) in
    a globally optimal manner with respect to the association costs.

    Returns a dictionary of associations of the form:
    {'lhs': {lhs_instance: rhs_instance}, 'rhs': {rhs_instance: lhs_instance}}

    In case of a nonmatch, the instance is associated with None.

    candidates_lhs -- iterable of candidate instances
    candidates_rhs -- iterable of candidate instances
    match_cost_function -- f(candidate_lhs, candidate_lhs) :: instance -> instance -> number
    nonmatch_cost_function -- f(candidate_lhs_or_rhs) :: instance -> number
    cost_threshold -- only consider associations below this threshold (runtime tuning parameter); applies only to
      match costs
    
    '''
    if candidates_lhs or candidates_rhs: # at least one set has to be non-empty; else, ilp would be rejected by solver
        graph = _construct_match_graph(candidates_lhs, candidates_rhs, match_cost_function, nonmatch_cost_function, cost_threshold)
        ilp, variables = _formulate_integer_linear_program( graph )
        ilp.solve(_pulp.GLPK(msg=False))
        assoc = _formulate_associations( graph, variables )
    else:
        assoc = {'lhs': {}, 'rhs': {}}
    return assoc
