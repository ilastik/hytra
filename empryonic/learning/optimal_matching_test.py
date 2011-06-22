#
# (c) Bernhard X. Kausler, 2010
#
import unittest
from unittest import TestCase
import optimal_matching as om


class TestCostThreshold( TestCase ):
    def test_vertices_without_edges( self ):
        no_edges = om.optimal_matching(['a'], ['b'], lambda x,y: 20, lambda x: 15, 10)
        self.assertEqual({'lhs': {'a': None}, 'rhs': {'b': None}}, no_edges)
    
    def test_basic_behaviour( self ):
        below_threshold = om.optimal_matching(['a'], ['b'], lambda x,y: 10, lambda x: 15, 11)
        self.assertEqual({'lhs': {'a': 'b'}, 'rhs': {'b': 'a'}}, below_threshold)
        above_threshold = om.optimal_matching(['a'], ['b'], lambda x,y: 10, lambda x: 15, 9)
        self.assertEqual({'lhs': {'a': None}, 'rhs': {'b': None}}, above_threshold)
        without_threshold = om.optimal_matching(['a'], ['b'], lambda x,y: 10, lambda x: 15, None)
        self.assertEqual({'lhs': {'a': 'b'}, 'rhs': {'b': 'a'}}, below_threshold)

class TestGlobalOptimality( TestCase ):
    # A simple example, that cannot be matched correctly by greedy algorithms.
    def runTest( self ):
        def cost_fct(x, y):
            table = {'a': {}, 'b': {}}
            table['a']['d'] = 3
            table['a']['c'] = 1
            table['b']['c'] = 2
            table['b']['d'] = 5
            return table[x][y]
        assoc = om.optimal_matching(['a','b'], ['c', 'd'], cost_fct, lambda x: 10)
        expected = {'lhs': {'a': 'd', 'b': 'c'}, 'rhs': {'d': 'a', 'c': 'b'}}
        self.assertEqual(assoc, expected)
        
class TestEmptyCandidates( TestCase ):
    def test_empty_lhs(self):
        assoc = om.optimal_matching([], [1,2], lambda x,y: 0, lambda x: 0)
        expected = {'lhs': {}, 'rhs': {1: None, 2: None}}
        self.assertEqual(assoc, expected)
           
    def test_empty_rhs(self):
        assoc = om.optimal_matching([1,2], [], lambda x,y: 0, lambda x: 0)
        expected = {'lhs': {1: None, 2: None}, 'rhs': {}}
        self.assertEqual(assoc, expected)
    
    def test_both_empty(self):
        assoc = om.optimal_matching([], [], lambda x,y: 0, lambda x: 0)
        self.assertEqual(assoc, {'lhs': {}, 'rhs': {}})

        

if __name__ == "__main__":
    unittest.main()
