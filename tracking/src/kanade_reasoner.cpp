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

    // false positive hypotheses
    size_t count = 0;
    vector<double> fp_costs;
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      tracklet_idx_map_[n] = count;
      hyp2type_[count] = FP;
      fp2node_[count] = n;
      fp_costs.push_back(log(fp_potential_(traxel_map[n])));
      ++count;
    }
    
    ilp_ = new KanadeIlp( fp_costs.begin(), fp_costs.end());

    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      add_hypotheses(g, n);
    }
  }

  void Kanade::infer() {
    ilp_->solve();
  }

  void Kanade::conclude( HypothesesGraph& g ) {
    // add active property to graph and init nodes/arcs as active/inactive
    g.add(node_active()).add(arc_active());
    property_map<node_active, HypothesesGraph::base_graph>::type& active_nodes = g.get(node_active());
    property_map<arc_active, HypothesesGraph::base_graph>::type& active_arcs = g.get(arc_active());

    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      active_nodes.set(n, true);
    }
    for(HypothesesGraph::ArcIt a(g); a!=lemon::INVALID; ++a) {
      active_arcs.set(a, false);
    }


    // go thru all hyps
    for(size_t hyp = 0; hyp < ilp_->solution_size(); ++hyp) {
      if(ilp_->hypothesis_is_active( hyp )) {
	  // mark corresponding nodes/arcs as active
	  switch( hyp2type_[hyp] ) {
	  case TERM:
	    // do nothing, because all arcs are inactive by default
	    break;
	  case INIT:
	    // do nothing, because all arcs are inactive by default
	    break;
	  case TRANS:
	    active_arcs.set(trans2arc_[hyp], true); 
	    break;
	  case DIV:
	    active_arcs.set(div2arcs_[hyp].first, true);
	    active_arcs.set(div2arcs_[hyp].second, true);
	    break;
	  case FP:
	    active_nodes.set(fp2node_[hyp], false);
	    break;
	  default:
	    throw runtime_error("Kanade::conclude(): unknown hypothesis type");
	    break;
	  }
	}
    }
  }

  void Kanade::reset() {
    if(ilp_ != NULL) {
      delete ilp_;
      ilp_ = NULL;
    }

    tracklet_idx_map_.clear();
    hyp2type_.clear();
    fp2node_.clear();
    trans2arc_.clear();
    div2arcs_.clear();
  }

  Kanade& Kanade::add_hypotheses( const HypothesesGraph& g, const HypothesesGraph::Node& n) {
    size_t hyp = 0;
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    double cost = 0;

    // init hypothesis
    cost = log(ini_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[n]));
    hyp = ilp_->add_init_hypothesis( tracklet_idx_map_[n], cost ); 
    hyp2type_[hyp] = INIT;

    // collect and count outgoing arcs
    vector<HypothesesGraph::Arc> arcs; 
    size_t count = 0;
    for(HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
      arcs.push_back(a);
      ++count;
    }

    if(count == 0) {
      cost = log(term_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[n]));
      hyp = ilp_->add_term_hypothesis(tracklet_idx_map_[n], cost );
      hyp2type_[hyp] = TERM;
    } else if(count == 1) {
      cost = log(term_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[n]));
      hyp = ilp_->add_term_hypothesis( tracklet_idx_map_[n], cost );
      hyp2type_[hyp] = TERM;
      
      cost = log(link_potential_(traxel_map[n], traxel_map[g.target(arcs[0])])) + 0.5 * log(tp_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[g.target(arcs[0])]));
      hyp = ilp_->add_trans_hypothesis( tracklet_idx_map_[n], tracklet_idx_map_[g.target(arcs[0])], cost);
      hyp2type_[hyp] = TRANS;
      trans2arc_[hyp] = arcs[0];
    } else {
      cost = log(term_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[n]));
      hyp = ilp_->add_term_hypothesis( tracklet_idx_map_[n], cost );
      hyp2type_[hyp] = TERM;
      
      // translation hypotheses
      for(size_t i = 0; i < count; ++i) {
	cost = log(link_potential_(traxel_map[n], traxel_map[g.target(arcs[i])])) + 0.5 * log(tp_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[g.target(arcs[i])]));
	hyp = ilp_->add_trans_hypothesis( tracklet_idx_map_[n], tracklet_idx_map_[g.target(arcs[i])], cost);
	hyp2type_[hyp] = TRANS;
	trans2arc_[hyp] = arcs[i];
      }

      // division hypotheses
      for(size_t i = 0; i < count - 1; ++i) {
	for(size_t j = i; j < count; ++j) {

	  cost = log(div_potential_(traxel_map[n], traxel_map[g.target(arcs[i])], traxel_map[g.target(arcs[j])])) + 0.5 * log(tp_potential_(traxel_map[n])) + 0.5 * log(tp_potential_(traxel_map[g.target(arcs[i])])) * log(tp_potential_(traxel_map[g.target(arcs[j])]));
	  hyp = ilp_->add_div_hypothesis( tracklet_idx_map_[n], 
				    tracklet_idx_map_[g.target(arcs[i])],
				    tracklet_idx_map_[g.target(arcs[j])],
				    cost);
	  hyp2type_[hyp] = DIV;
	  div2arcs_[hyp] = pair<HypothesesGraph::Arc, HypothesesGraph::Arc>(arcs[i], arcs[j]);
	}
      }
    }

    return *this;
  }

} /* namespace Tracking */ 
