#define BOOST_TEST_MODULE kanade_reasoner_test

#include <vector>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "reasoning/kanade_reasoner.h"

using namespace Tracking;
using namespace std;
using namespace boost;

BOOST_AUTO_TEST_CASE( KanadeIlp_construction ) {
  KanadeIlp ilp(7);

}

BOOST_AUTO_TEST_CASE( KanadeIlp_example_problem ) {
  size_t n_tracklets = 7;
  KanadeIlp ilp(n_tracklets);
  ilp.add_init_hypothesis(0, 0.7);
  ilp.add_trans_hypothesis(0, 1, 0.8);
  ilp.add_trans_hypothesis(0, 2, 0.8);
  ilp.add_term_hypothesis(2, 0.8);
  ilp.add_init_hypothesis(3, 0.9);
  ilp.add_trans_hypothesis(3, 4, 0.5);
  ilp.add_trans_hypothesis(3, 5, 0.6);
  ilp.add_div_hypothesis(3, 4, 5, 0.6);
  ilp.add_trans_hypothesis(4, 6, 0.7);
  ilp.add_term_hypothesis(5, 0.8);
  ilp.add_term_hypothesis(6, 0.8);

  vector<int> solution = ilp.solve().get_solution();
  
  int expected_solution[] = {1,0,1,1,1,0,0,1,1,1,1,
                             0,1,0,0,0,0,0}; 
  BOOST_REQUIRE_EQUAL(solution.size(), 11 + n_tracklets);
  BOOST_REQUIRE_EQUAL_COLLECTIONS(solution.begin(), solution.end(),
		      expected_solution, expected_solution + 
		      (sizeof(expected_solution) / sizeof(int)));
} 
