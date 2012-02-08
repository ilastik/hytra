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
  /** ids will be assumed to lie consecutivly in the range 0 ... (n_tracklets-1) **/
  KanadeIlp( size_t n_tracklets );
  ~KanadeIlp();

  KanadeIlp& add_init_hypothesis(size_t id, double cost); // initialization
  KanadeIlp& add_term_hypothesis(size_t id, double cost); // termination
  KanadeIlp& add_trans_hypothesis(size_t from_id, size_t to_id, double cost); // translation
  KanadeIlp& add_div_hypothesis(size_t from_id, size_t to_id1, size_t to_id2, double cost); // division

  // this is not necessary, since all tracklets can be false positives...
  KanadeIlp& add_fp_hypothesis(size_t id, double cost); // false positive

  KanadeIlp& solve();
  std::vector<int> get_solution();

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
