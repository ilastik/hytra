#include "reasoning/kanade_reasoner.h"

using namespace std;

namespace Tracking {

////
//// class KanadeIlp
////
  KanadeIlp::KanadeIlp( size_t n_tracklets ) : n_tracklets_(n_tracklets) {
    model_ = IloModel(env_);
    IloObjective obj_ = IloMaximize( env_ );
    //model_.add(obj_);
    //IloBoolVarArray bla;
    
 
    //IloRange c_(env_, 1, 1);
    //model_.add(c_);

    //IloNumColumn col = obj_(0.7) + c_(0);

    //x_.add(IloBoolVar( col ));
    //
  }

  KanadeIlp::~KanadeIlp() {
    env_.end();
  }


  KanadeIlp& KanadeIlp::add_init_hypothesis(size_t id, double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_term_hypothesis(size_t id, double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_trans_hypothesis(size_t from_id, size_t to_id, double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_div_hypothesis(size_t from_id, size_t to_id1, size_t to_id2, double cost) {
    return *this;
  }



  KanadeIlp& KanadeIlp::add_fp_hypothesis(size_t id, double cost) {
    
    return *this;
  }



  KanadeIlp& KanadeIlp::solve() {
    return *this;
  }

  vector<int> KanadeIlp::get_solution() {
    return vector<int>();
  }

  


void Kanade::formulate( const HypothesesGraph& hypotheses ) {
}



void Kanade::infer() {
}


void Kanade::conclude( HypothesesGraph& g ) {
}



} /* namespace Tracking */ 
