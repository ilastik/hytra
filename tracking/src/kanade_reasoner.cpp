#include <cassert>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include "reasoning/kanade_reasoner.h"

using namespace std;

namespace Tracking {

////
//// class KanadeIlp
////
  KanadeIlp::~KanadeIlp() {
    env_.end();
  }

  size_t KanadeIlp::add_init_hypothesis(size_t idx, double cost) {
    assert(idx < n_tracklets_);
    x_.add(IloBoolVar( obj_(cost) + c_[n_tracklets_ + idx](1) ));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_term_hypothesis(size_t idx, double cost) {
    assert(idx < n_tracklets_);
    x_.add(IloBoolVar( obj_(cost) + c_[idx](1) ));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_trans_hypothesis(size_t from_idx, size_t to_idx, double cost) {
    assert(from_idx < n_tracklets_);
    assert(to_idx < n_tracklets_);
    x_.add(IloBoolVar( obj_(cost) + c_[from_idx](1) + c_[n_tracklets_ + to_idx](1) ));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_div_hypothesis(size_t from_idx, size_t to_idx1, size_t to_idx2, double cost) {
    assert(from_idx < n_tracklets_);
    assert(to_idx1 < n_tracklets_);
    assert(to_idx2 < n_tracklets_);
    x_.add(IloBoolVar( obj_(cost) + c_[from_idx](1) + c_[n_tracklets_ + to_idx1](1) + c_[n_tracklets_ + to_idx2](1) ));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_fp_hypothesis(size_t idx, double cost) {
    assert(idx < n_tracklets_);
    x_.add(IloBoolVar( obj_(cost) + c_[idx](1) + c_[n_tracklets_ + idx](1)));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  KanadeIlp& KanadeIlp::solve() {
    // assemble model
    model_.add(obj_);
    model_.add(x_);
    model_.add(c_);

    // solve
    cplex_.extract(model_);
    if( !cplex_.solve() ) {
      throw runtime_error("KanadeIlp::solve(): cplex.solve() failed");
    }
    env_.out() << "Kanade solution status = " << cplex_.getStatus() << "\n";
    env_.out() << "Kanade objective value: " << cplex_.getObjValue() << "\n"; 
    cplex_.getValues(sol_, x_);
    //env_.out() << "Kanade solution: " << sol_ << "\n";

    return *this;
  }

  size_t KanadeIlp::solution_size() {
    size_t size = static_cast<size_t>(sol_.getSize());
    assert(size == n_hypotheses_);
    return size;
  }
  
  bool KanadeIlp::hypothesis_is_active( size_t idx ) {
    assert(idx < n_hypotheses_);
    return static_cast<bool>(sol_[idx]);
  }



  ////
  //// class Kanade
  ////
  Kanade::~Kanade() {
    if(ilp_ != NULL) {
      delete ilp_;
      ilp_ = NULL;
    }
  }
  
  void Kanade::formulate( const HypothesesGraph& g ) {
    reset();

    // flase positive hypotheses
    size_t count = 0;
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      tracklet_idx_map_[n] = count;
      hyp2type_[count] = FP;
      fp2node_[count] = n;
      ++count;
    }
    vector<double> fp_costs(lemon::countNodes( g ), log(misdetection_rate_));
    
    ilp_ = new KanadeIlp( fp_costs.begin(), fp_costs.end());

    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      add_hypotheses(g, n);
    }
  }

  void Kanade::infer() {
}

  void Kanade::conclude( HypothesesGraph& /*g*/ ) {
}

  void Kanade::reset() {
    if(ilp_ != NULL) {
      delete ilp_;
      ilp_ = NULL;
    }
  }

  Kanade& Kanade::add_hypotheses( const HypothesesGraph& g, const HypothesesGraph::Node& n) {
    double COST = 100.2;
    size_t hyp = 0;

    // init hypothesis
    hyp = ilp_->add_init_hypothesis( tracklet_idx_map_[n], COST ); 
    hyp2type_[hyp] = INIT;

    // collect and count outgoing arcs
    vector<HypothesesGraph::Arc> arcs; 
    size_t count = 0;
    for(HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
      arcs.push_back(a);
      ++count;
    }

    if(count == 0) {
      hyp = ilp_->add_term_hypothesis(tracklet_idx_map_[n], COST );
      hyp2type_[hyp] = TERM;
    } else if(count == 1) {
      hyp = ilp_->add_term_hypothesis( tracklet_idx_map_[n], COST );
      hyp2type_[hyp] = TERM;
      
      hyp = ilp_->add_trans_hypothesis( tracklet_idx_map_[n], tracklet_idx_map_[g.target(arcs[0])], COST);
      hyp2type_[hyp] = TRANS;
      trans2arc_[hyp] = arcs[0];
    } else {
      hyp = ilp_->add_term_hypothesis( tracklet_idx_map_[n], COST );
      hyp2type_[hyp] = TERM;
      
      // translation hypotheses
      for(size_t i = 0; i < count; ++i) {
	hyp = ilp_->add_trans_hypothesis( tracklet_idx_map_[n], tracklet_idx_map_[g.target(arcs[i])], COST);
	hyp2type_[hyp] = TRANS;
	trans2arc_[hyp] = arcs[i];
      }

      // division hypotheses
      for(size_t i = 0; i < count - 1; ++i) {
	for(size_t j = i; j < count; ++j) {
	  hyp = ilp_->add_div_hypothesis( tracklet_idx_map_[n], 
				    tracklet_idx_map_[g.target(arcs[i])],
				    tracklet_idx_map_[g.target(arcs[j])],
				    COST);
	  hyp2type_[hyp] = DIV;
	  div2arcs_[hyp] = pair<HypothesesGraph::Arc, HypothesesGraph::Arc>(arcs[i], arcs[j]);
	}
      }
    }

    return *this;
  }

} /* namespace Tracking */ 
