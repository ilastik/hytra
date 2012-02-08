#include <cassert>
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
    IloBoolVar( obj_(cost) + c_[n_tracklets_ + idx](1) );
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_term_hypothesis(size_t idx, double cost) {
    assert(idx < n_tracklets_);
    IloBoolVar( obj_(cost) + c_[idx](1) );
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_trans_hypothesis(size_t from_idx, size_t to_idx, double cost) {
    assert(from_idx < n_tracklets_);
    assert(to_idx < n_tracklets_);
    IloBoolVar( obj_(cost) + c_[from_idx](1) + c_[n_tracklets_ + to_idx](1) );
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_div_hypothesis(size_t from_idx, size_t to_idx1, size_t to_idx2, double cost) {
    assert(from_idx < n_tracklets_);
    assert(to_idx1 < n_tracklets_);
    assert(to_idx2 < n_tracklets_);
    IloBoolVar( obj_(cost) + c_[from_idx](1) + c_[n_tracklets_ + to_idx1](1) + c_[n_tracklets_ + to_idx1](1) );
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  size_t KanadeIlp::add_fp_hypothesis(size_t idx, double cost) {
    assert(idx < n_tracklets_);
    IloBoolVar( obj_(cost) + c_[idx](1) + c_[n_tracklets_ + idx](1));
    n_hypotheses_ += 1;
    return n_hypotheses_ - 1;
  }

  KanadeIlp& KanadeIlp::solve() {
    return *this;
  }

  size_t KanadeIlp::solution_size() {
    return 0;
  }
  
  bool KanadeIlp::hypothesis_is_active( size_t idx ) {
    assert(idx < n_hypotheses_);
    return false;
  }



  void Kanade::formulate( const HypothesesGraph& /*hypotheses*/ ) {
}



void Kanade::infer() {
}


  void Kanade::conclude( HypothesesGraph& /*g*/ ) {
}



} /* namespace Tracking */ 
