#include "reasoning/kanade_reasoner.h"

using namespace std;

namespace Tracking {

////
//// class KanadeIlp
////
  KanadeIlp& KanadeIlp::add_init_hypothesis(double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_term_hypothesis(double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_trans_hypothesis(double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_div_hypothesis(double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_fp_hypothesis(double cost) {
    return *this;
  }



void Kanade::formulate( const HypothesesGraph& hypotheses ) {
}



void Kanade::infer() {
}


void Kanade::conclude( HypothesesGraph& g ) {
}



} /* namespace Tracking */ 
