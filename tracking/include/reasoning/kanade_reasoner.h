#ifndef KANADE_REASONER_H
#define KANADE_REASONER_H

#include <iterator>
#include <vector>
#include <ilcplex/ilocplex.h>
#include "reasoning/reasoner.h"

namespace Tracking {
class Traxel;
class HypothesesGraph;

class KanadeIlp {
 public:
  /** idxs will be assumed to lie consecutivly in the range 0 ... (n_tracklets-1) **/
  template <typename iterator>
    KanadeIlp( iterator begin_fp_cost, iterator end_fp_cost );
  ~KanadeIlp();

  size_t add_init_hypothesis(size_t idx, double cost); // initialization
  size_t add_term_hypothesis(size_t idx, double cost); // termination
  size_t add_trans_hypothesis(size_t from_idx, size_t to_idx, double cost); // translation
  size_t add_div_hypothesis(size_t from_idx, size_t to_idx1, size_t to_idx2, double cost); // division

  KanadeIlp& solve();
  size_t solution_size();
  bool hypothesis_is_active( size_t hypothesis_id );

 protected:
  size_t add_fp_hypothesis(size_t idx, double cost); // false positive

 private:
  size_t n_tracklets_;
  size_t n_hypotheses_;
  
  IloEnv env_;
  IloModel model_;
  IloObjective obj_;
  IloBoolVarArray x_;
  IloRangeArray c_;
};



class Kanade : public Reasoner {
    public:
 Kanade() : ilp_(NULL) {}

    virtual void formulate( const HypothesesGraph& );
    virtual void infer();
    virtual void conclude( HypothesesGraph& );

    private:
    // copy and assingment have to be implemented, yet
    Kanade(const Kanade&) {};
    Kanade& operator=(const Kanade&) { return *this;};

    KanadeIlp* ilp_;
};



/******************/
/* Implementation */
/******************/

 template<typename iterator>
   KanadeIlp::KanadeIlp( iterator begin, iterator end ) : n_tracklets_(distance(begin, end)), n_hypotheses_(0) {
    model_ = IloModel(env_);
    obj_ = IloMaximize( env_ );

    // 2 * n constraints: sum == 1
    c_ = IloRangeArray( env_, 2 * n_tracklets_, 1, 1);

    // for every tracklet: add a false positve hypothesis
    size_t idx = 0, hyp_idx = 0;
    for(iterator cost_it = begin; cost_it != end; ++cost_it) {
      hyp_idx = add_fp_hypothesis(idx, *cost_it);
      assert(hyp_idx == idx);
      ++idx;
    }
  }

} /* namespace Tracking */

#endif /* KANADE_REASONER_H */
