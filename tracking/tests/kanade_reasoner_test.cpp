#define BOOST_TEST_MODULE kanade_reasoner_test

#include <vector>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "reasoning/kanade_reasoner.h"

using namespace Tracking;
using namespace std;
using namespace boost;

BOOST_AUTO_TEST_CASE( KanadeIlp_example_problem ) {
  double fp_costs[] = { 0.2,0.8,0.1,0.1,0.1,0.1,0.1 };
  size_t n_tracklets = sizeof(fp_costs)/sizeof(double);
  BOOST_REQUIRE_EQUAL( n_tracklets, 7);

  KanadeIlp ilp(fp_costs, fp_costs + n_tracklets);

  size_t hyp_idx = 0;
  hyp_idx = ilp.add_init_hypothesis(0, 0.7);
  BOOST_CHECK_EQUAL(hyp_idx, 7);
  ilp.add_trans_hypothesis(0, 1, 0.8);
  ilp.add_trans_hypothesis(0, 2, 0.8);
  ilp.add_term_hypothesis(2, 0.8);
  ilp.add_init_hypothesis(3, 0.9);
  ilp.add_trans_hypothesis(3, 4, 0.5);
  ilp.add_trans_hypothesis(3, 5, 0.6);
  ilp.add_div_hypothesis(3, 4, 5, 0.6);
  ilp.add_trans_hypothesis(4, 6, 0.7);
  ilp.add_term_hypothesis(5, 0.8);
  hyp_idx = ilp.add_term_hypothesis(6, 0.8);
  BOOST_CHECK_EQUAL(hyp_idx, 17);

  ilp.solve();
  bool expected[] = {
    false,true,false,false,false,false,false, // false positives
    true,false,true,true,true,false,false,true,true,true,true};

  BOOST_REQUIRE_EQUAL(ilp.solution_size(), 11 + n_tracklets);
  for(size_t i = 0; i < ilp.solution_size(); ++i) {
    BOOST_CHECK_EQUAL(ilp.hypothesis_is_active( i ), expected[i]);
  }
} 
