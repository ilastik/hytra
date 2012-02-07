#ifndef KANADE_REASONER_H
#define KANADE_REASONER_H

#include <ilcplex/ilocplex.h>
#include "reasoning/reasoner.h"

namespace Tracking {
class Traxel;
class HypothesesGraph;

class KanadeIlp {
 public:
  KanadeIlp( unsigned int n_tracklets );
  ~KanadeIlp();

  KanadeIlp& add_init_hypothesis(double cost); // initialization
  KanadeIlp& add_term_hypothesis(double cost); // termination
  KanadeIlp& add_trans_hypothesis(double cost); // translation
  KanadeIlp& add_div_hypothesis(double cost); // division
  KanadeIlp& add_fp_hypothesis(double cost); // false positive

 private:
  unsigned int n_tracklets_;
  
  IloEnv env_;
  IloModel model_;
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
