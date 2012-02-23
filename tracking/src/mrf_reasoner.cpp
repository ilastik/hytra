#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/datastructures/marray/marray.hxx>

#include "graphical_model.h"
#include "hypotheses.h"
#include "log.h"
#include "reasoning/mrf_reasoner.h"
#include "traxels.h"

//#include <ostream>

using namespace std;

namespace Tracking {
  SingleTimestepTraxelMrf::~SingleTimestepTraxelMrf() {
   if(mrf_ != NULL) {
	delete mrf_;
	mrf_ = NULL;
    }
    if(optimizer_ != NULL) {
	delete optimizer_;
	optimizer_ = NULL;
    }
  }

double SingleTimestepTraxelMrf::forbidden_cost() const {
    return forbidden_cost_;
}

bool SingleTimestepTraxelMrf::with_constraints() const {
    return with_constraints_;
}

void SingleTimestepTraxelMrf::formulate( const HypothesesGraph& hypotheses ) {
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::formulate: entered";
    reset();
    mrf_ = new OpengmMrf();

    LOG(logDEBUG) << "SingleTimestepTraxelMrf::formulate: add_detection_nodes";
    add_detection_nodes( hypotheses );
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::formulate: add_transition_nodes";
    add_transition_nodes( hypotheses );
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::formulate: add_finite_factors";
    add_finite_factors( hypotheses );


    typedef opengm::LPCplex<OpengmMrf::ogmGraphicalModel, OpengmMrf::ogmAccumulator> cplex_optimizer;
    cplex_optimizer::Parameter param;
    param.verbose_ = true;
    param.integerConstraint_ = true;
    param.epGap_ = 0.05;

    OpengmMrf::ogmGraphicalModel* model = mrf_->Model();
    optimizer_ = new cplex_optimizer(*model, param);
    if(with_constraints_) {
	if(constraints_as_infinite_energy_ == true) {
	    throw "SingleTimestepTraxelMrf::formulate(): constraints_as_infite_energy not supported yet.";
	}
	LOG(logDEBUG) << "SingleTimestepTraxelMrf::formulate: add_constraints";
	add_constraints( hypotheses );
    }
}



void SingleTimestepTraxelMrf::infer() {
    opengm::InferenceTermination status = optimizer_->infer();
    if(status != opengm::NORMAL) {
        throw std::runtime_error("GraphicalModel::infer(): optimizer terminated unnormally");
    }
}


void SingleTimestepTraxelMrf::conclude( HypothesesGraph& g ) {
    // extract solution from optimizer
    vector<OpengmMrf::ogmInference::LabelType> solution;
    opengm::InferenceTermination status = optimizer_->arg(solution);
    if(status != opengm::NORMAL) {
	throw runtime_error("GraphicalModel::infer(): solution extraction terminated unnormally");
    }

    // add 'active' properties to graph
    g.add(node_active()).add(arc_active());
    property_map<node_active, HypothesesGraph::base_graph>::type& active_nodes = g.get(node_active());
    property_map<arc_active, HypothesesGraph::base_graph>::type& active_arcs = g.get(arc_active());

    // write state after inference into 'active'-property maps
    for(std::map<HypothesesGraph::Node, size_t>::const_iterator it = node_map_.begin(); it != node_map_.end(); ++it) {
	bool state = false;
	if(solution[it->second] == 1) state = true;
	active_nodes.set(it->first, state);
    }
    for(std::map<HypothesesGraph::Arc, size_t>::const_iterator it = arc_map_.begin(); it != arc_map_.end(); ++it) {
	bool state = false;
	if(solution[it->second] == 1) state = true;
	active_arcs.set(it->first, state);
    }
}

  const OpengmMrf* SingleTimestepTraxelMrf::get_graphical_model() const {
    return mrf_;
  }

  const std::map<HypothesesGraph::Node, size_t>& SingleTimestepTraxelMrf::get_node_map() const {
    return node_map_;
  }

  const std::map<HypothesesGraph::Arc, size_t>& SingleTimestepTraxelMrf::get_arc_map() const {
    return arc_map_;
  }

void SingleTimestepTraxelMrf::reset() {
    if(mrf_ != NULL) {
	delete mrf_;
	mrf_ = NULL;
    }
    if(optimizer_ != NULL) {
	delete optimizer_;
	optimizer_ = NULL;
    }
    node_map_.clear();
    arc_map_.clear();
}

void SingleTimestepTraxelMrf::add_detection_nodes( const HypothesesGraph& g) {
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
	mrf_->Model()->addVariable(2);
	node_map_[n] = mrf_->Model()->numberOfVariables() - 1; 
    }
}
void SingleTimestepTraxelMrf::add_transition_nodes( const HypothesesGraph& g) {
    for(HypothesesGraph::ArcIt a(g); a!=lemon::INVALID; ++a) {
	mrf_->Model()->addVariable(2);
	arc_map_[a] = mrf_->Model()->numberOfVariables() - 1; 
    }
}

void SingleTimestepTraxelMrf::add_finite_factors( const HypothesesGraph& g) {
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_finite_factors: entered";
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());		
    ////
    //// add detection factors
    ////
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_finite_factors: add detection factors";
   for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
	size_t vi[] = {node_map_[n]};						// node index
	const size_t shape[]={mrf_->Model()->numberOfStates(*vi)}; 		// graphicalmodel_factor.hxx (l.950)
	OpengmMrf::ExplicitFunctionType f(shape,shape+1);			
	f(0) = non_detection_(traxel_map[n]);
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_finite_factors: non_detection energy: "<< f(0);
	f(1) = detection_(traxel_map[n]);
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_finite_factors: detection energy: "<< f(1);
	OpengmMrf::FunctionIdentifier id=mrf_->Model()->addFunction(f);	
	mrf_->Model()->addFactor(id,vi,vi+1); 
    }

    ////
    //// add transition factors
    ////
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
      add_outgoing_factor(g, n);
      add_incoming_factor(g, n);
    }
}	    

namespace {
    inline size_t cplex_id(size_t opengm_id) {
	return 2*opengm_id + 1;
    }
}

void SingleTimestepTraxelMrf::add_constraints( const HypothesesGraph& g) {
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_constraints: entered";
    typedef opengm::LPCplex<OpengmMrf::ogmGraphicalModel, OpengmMrf::ogmAccumulator> cplex;
    ////
    //// outgoing transitions
    ////
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_constraints: outgoing transitions";
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
	// couple detection and transitions
	for(HypothesesGraph::OutArcIt a(g, n); a!=lemon::INVALID; ++a) {
	    couple(n, a);
	}
	// couple transitions
	vector<size_t> cplex_idxs;
	for(HypothesesGraph::OutArcIt a(g, n); a!=lemon::INVALID; ++a) {
	    cplex_idxs.push_back(cplex_id(arc_map_[a]));
	}
	vector<int> coeffs(cplex_idxs.size(), 1);
	// 0 <= 1*transition + ... + 1*transition <= 2
	dynamic_cast<cplex*>(optimizer_)->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0, 2);
    }

    ////
    //// incoming transitions
    ////
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_constraints: incoming transitions";
    for(HypothesesGraph::NodeIt n(g); n!=lemon::INVALID; ++n) {
	// couple detection and transitions
	for(HypothesesGraph::InArcIt a(g, n); a!=lemon::INVALID; ++a) {
	    couple(n, a);
	}
	    
	// couple transitions
	vector<size_t> cplex_idxs;
	for(HypothesesGraph::InArcIt a(g, n); a!=lemon::INVALID; ++a) {
	    cplex_idxs.push_back(cplex_id(arc_map_[a]));
	}
	vector<int> coeffs(cplex_idxs.size(), 1);
	// 0 <= 1*transition + ... + 1*transition <= 1
	dynamic_cast<cplex*>(optimizer_)->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin(), 0, 1);
    }
}

void SingleTimestepTraxelMrf::couple(HypothesesGraph::Node& n, HypothesesGraph::Arc& a) {
	    typedef opengm::LPCplex<OpengmMrf::ogmGraphicalModel, OpengmMrf::ogmAccumulator> cplex;
    	    vector<size_t> cplex_idxs; 
	    cplex_idxs.push_back(cplex_id(node_map_[n]));
	    cplex_idxs.push_back(cplex_id(arc_map_[a]));
	    vector<int> coeffs;
	    coeffs.push_back(1);
	    coeffs.push_back(-1);
	    // 0 <= 1*detection - 1*transition <= 1
	    dynamic_cast<cplex*>(optimizer_)->addConstraint(cplex_idxs.begin(), cplex_idxs.end(), coeffs.begin() , 0, 1);
}

  void SingleTimestepTraxelMrf::add_outgoing_factor( const HypothesesGraph& g, const HypothesesGraph::Node& n ) {
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_outgoing_factor(): entered";
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    // collect and count outgoing arcs
    vector<HypothesesGraph::Arc> arcs; 
    vector<size_t> vi; 		// opengm variable indeces
    vi.push_back(node_map_[n]); // first detection node, remaining will be transition nodes
    int count = 0;
    for(HypothesesGraph::OutArcIt a(g, n); a != lemon::INVALID; ++a) {
      arcs.push_back(a);
      vi.push_back(arc_map_[a]);
      ++count;
    }

    // safeguard, if something in lemon changes; opengm expects a monotonic increasing sequence
    if(!(mrf_->Model()->isValidIndexSequence(vi.begin(), vi.end()))) {
      throw std::runtime_error("SingleTimestepTraxelMrf::add_outgoing_factor(): invalid index sequence");
    }

    // construct factor
    if(count == 0) {
      // build value table
      typedef OpengmMrf::ExplicitFunctionType table_t;
      size_t table_dim = 1; 		// only one detection var
      vector<size_t> shape(table_dim, 2);
      table_t table(shape.begin(), shape.end(), forbidden_cost_);
      std::vector<size_t> coords;
      size_t index = 0;
      table_t::iterator element(table);

	// opportunity
        coords = std::vector<size_t>(table_dim, 0); 		// (0)
        table.coordinatesToIndex(coords.begin(), index);
        element[index] = opportunity_cost_;
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: opportunity: "<< element[index];

	// disappearance 
     	coords = std::vector<size_t>(table_dim, 0);
   	coords[0] = 1; 						// (1)
        table.coordinatesToIndex(coords.begin(), index);
        element[index] = disappearance_(traxel_map[n]);
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: disappearance: "<< element[index];

	OpengmMrf::FunctionIdentifier id=mrf_->Model()->addFunction(table);	
	mrf_->Model()->addFactor(id,vi.begin(),vi.end());

    } else if(count == 1) {
      // no division possible
      typedef OpengmMrf::ExplicitFunctionType table_t;
      size_t table_dim = 2; 		// detection var + 1 * transition var
      vector<size_t> shape(table_dim, 2);
      table_t table(shape.begin(), shape.end(), forbidden_cost_);
      std::vector<size_t> coords;
      size_t index = 0;
      table_t::iterator element(table);

      // opportunity configuration
      coords = std::vector<size_t>(table_dim, 0); // (0,0)
      table.coordinatesToIndex(coords.begin(), index);
      element[index] = opportunity_cost_;
      LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: opportunity: "<< element[index];

      // disappearance configuration
      coords = std::vector<size_t>(table_dim, 0);
      coords[0] = 1; // (1,0)
      table.coordinatesToIndex(coords.begin(), index);
      element[index] = disappearance_(traxel_map[n]);
      LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: disappearance: "<< element[index];

      // move configurations
      coords = std::vector<size_t>(table_dim, 1);
      // (1,1)
	table.coordinatesToIndex(coords.begin(), index);
	element[index] = move_(traxel_map[n], traxel_map[g.target(arcs[0])]);
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: move: "<< element[index];

      // add factor
	OpengmMrf::FunctionIdentifier id=mrf_->Model()->addFunction(table);	
	mrf_->Model()->addFactor(id,vi.begin(),vi.end());
    } else {
      // build value table
      typedef OpengmMrf::ExplicitFunctionType table_t;
      size_t table_dim = count + 1; 		// detection var + n * transition var
      vector<size_t> shape(table_dim, 2);
      table_t table(shape.begin(), shape.end(), forbidden_cost_);
      std::vector<size_t> coords;
      size_t index = 0;
      table_t::iterator element(table);

      // opportunity configuration
      coords = std::vector<size_t>(table_dim, 0); // (0,0,...,0)
      table.coordinatesToIndex(coords.begin(), index);
      element[index] = opportunity_cost_;
      LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: opportunity: "<< element[index];

      // disappearance configuration
      coords = std::vector<size_t>(table_dim, 0);
      coords[0] = 1; // (1,0,...,0)
      table.coordinatesToIndex(coords.begin(), index);
      element[index] = disappearance_(traxel_map[n]);
      LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: disappearance: "<< element[index];

      // move configurations
      coords = std::vector<size_t>(table_dim, 0);
      coords[0] = 1;
      // (1,0,0,0,1,0,0)
      for(size_t i = 1; i < table_dim; ++i) {
	coords[i] = 1; 
	table.coordinatesToIndex(coords.begin(), index);
	element[index] = move_(traxel_map[n], traxel_map[g.target(arcs[i-1])]);
	LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: move: "<< element[index];
	coords[i] = 0; // reset coords
      }
      
      // division configurations
      coords = std::vector<size_t>(table_dim, 0);
      coords[0] = 1;
      // (1,0,0,1,0,1,0,0) 
      for(unsigned int i = 1; i < table_dim - 1; ++i) {
	for(unsigned int j = i+1; j < table_dim; ++j) {
	  coords[i] = 1;
	  coords[j] = 1;
	  table.coordinatesToIndex(coords.begin(), index);
	  element[index] = division_(traxel_map[n],
				 traxel_map[g.target(arcs[i-1])],
				 traxel_map[g.target(arcs[j-1])]
				 );
	  LOG(logDEBUG3) << "SingleTimestepTraxelMrf::add_outgoing_factors: division: "<< element[index];
	  // reset
  	  coords[i] = 0;
	  coords[j] = 0;
	}
      }
      // add factor
	OpengmMrf::FunctionIdentifier id=mrf_->Model()->addFunction(table);	
	mrf_->Model()->addFactor(id,vi.begin(),vi.end());
    }   
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_outgoing_factor(): leaving";
}

  void SingleTimestepTraxelMrf::add_incoming_factor( const HypothesesGraph& g, const HypothesesGraph::Node& n ) {
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_incoming_factor(): entered";
    property_map<node_traxel, HypothesesGraph::base_graph>::type& traxel_map = g.get(node_traxel());
    // collect and count incoming arcs
    vector<size_t> vi; // opengm variable indeces
    int count = 0;
    for(HypothesesGraph::InArcIt a(g, n); a != lemon::INVALID; ++a) {
      vi.push_back(arc_map_[a]);
      ++count;
    }
    vi.push_back(node_map_[n]); 

    // hypothesis graph is built in a way, that this is a valid index sequence
    std::reverse(vi.begin(), vi.end());
    if(!(mrf_->Model()->isValidIndexSequence(vi.begin(), vi.end()))) {
      throw std::runtime_error("SingleTimestepTraxelMrf::add_incoming_factor(): invalid index sequence");
    }
    
    //// construct factor
    // build value table
      typedef OpengmMrf::ExplicitFunctionType table_t;
    size_t table_dim = count + 1; // detection var + n * transition var
    vector<size_t> shape(table_dim, 2);
    table_t table(shape.begin(), shape.end(), forbidden_cost_);
    std::vector<size_t> coords;
    size_t index = 0;
    table_t::iterator element(table);

    // allow opportunity configuration
    coords = std::vector<size_t>(table_dim, 0); // (0,0,...,0)
    table.coordinatesToIndex(coords.begin(), index);
    element[index] = 0;

    // appearance configuration
    coords = std::vector<size_t>(table_dim, 0);
    coords[0] = 1; // (1,0,...,0)
    table.coordinatesToIndex(coords.begin(), index);
    element[index] = appearance_(traxel_map[n]);
    
    // allow move configurations
    coords = std::vector<size_t>(table_dim, 0);
    coords[0] = 1;
    // (1,0,0,0,1,0,0)
    for(size_t i = 1; i < table_dim; ++i) {
      coords[i] = 1; 
      table.coordinatesToIndex(coords.begin(), index);
      element[index] = 0;
      coords[i] = 0; // reset coords
    }

    // add factor
    OpengmMrf::FunctionIdentifier id=mrf_->Model()->addFunction(table);
    mrf_->Model()->addFactor(id,vi.begin(),vi.end());
    LOG(logDEBUG) << "SingleTimestepTraxelMrf::add_incoming_factor(): leaving";
  }



} /* namespace Tracking */ 
