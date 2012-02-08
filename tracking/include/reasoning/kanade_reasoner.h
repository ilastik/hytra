#ifndef KANADE_REASONER_H
#define KANADE_REASONER_H

#include <vector>
#include <ilcplex/ilocplex.h>
#include "reasoning/reasoner.h"

namespace Tracking {
class Traxel;
class HypothesesGraph;

class KanadeIlp {
 public:
  /** idxs will be assumed to lie consecutivly in the range 0 ... (n_tracklets-1) **/
  KanadeIlp( size_t n_tracklets );
  ~KanadeIlp();

  size_t add_init_hypothesis(size_t idx, double cost); // initialization
  size_t add_term_hypothesis(size_t idx, double cost); // termination
  size_t add_trans_hypothesis(size_t from_idx, size_t to_idx, double cost); // translation
  size_t add_div_hypothesis(size_t from_idx, size_t to_idx1, size_t to_idx2, double cost); // division

  KanadeIlp& solve();
  size_t solution_size();
  bool hypothesis_is_active( size_t hypothesis_id );

 private:
  size_t n_tracklets_;
  
  IloEnv env_;
  IloModel model_;
  IloBoolVarArray x_;
  IloObjective obj_;
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

} /* namespace Tracking */
#endif /* KANADE_REASONER_H */
