#ifndef KANADE_REASONER_H
#define KANADE_REASONER_H

#include "ilp_construction.h"
#include "reasoning/reasoner.h"

namespace Tracking {
class Traxel;
class HypothesesGraph;

class KanadeIlp : public IntegerLinearProgram {
 public:
  KanadeIlp& add_init_hypothesis(double cost); // initialization
  KanadeIlp& add_term_hypothesis(double cost); // termination
  KanadeIlp& add_trans_hypothesis(double cost); // translation
  KanadeIlp& add_div_hypothesis(double cost); // division
  KanadeIlp& add_fp_hypothesis(double cost); // false positive
};



class Kanade : public Reasoner {
    public:
    virtual void formulate( const HypothesesGraph& );
    virtual void infer();
    virtual void conclude( HypothesesGraph& );

    private:
    // copy and assingment have to be implemented, yet
    Kanade(const Kanade&) {};
    Kanade& operator=(const Kanade&) { return *this;};

    KanadeIlp ilp_;
};

} /* namespace Tracking */
#endif /* KANADE_REASONER_H */
