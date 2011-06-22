#include <algorithm>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <ANN.h>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_array.hpp>

#include "ilp_construction.h"
#include "traxels.h"


namespace Tracking {

using boost::shared_ptr;
using boost::scoped_array;
using std::advance;
using std::equal;
using std::vector;
using std::map;
using std::pair;
using std::string;

using namespace Tracking::legacy;


///
/// class IntegerLinearProgram
///
bool IntegerLinearProgram::is_consistent() const {
    bool consistent = false;
    if(
	costs.size() == static_cast<unsigned int>(nVars) &&
	rhs.size() == static_cast<unsigned int>(nConstr) &&
	matbeg.size() == static_cast<unsigned int>(nVars) &&
	matcnt.size() == static_cast<unsigned int>(nVars) &&
	matind.size() == static_cast<unsigned int>(nNonZero) &&
	matval.size() == static_cast<unsigned int>(nNonZero) 
    ) {
	consistent = true;
    }
    return consistent;
}



bool IntegerLinearProgram::operator==(const IntegerLinearProgram& other) const {
    bool same = false;
    if(
	other.nVars == nVars &&
	other.nConstr == nConstr &&
	other.nNonZero == nNonZero &&
	other.costs.size() == other.costs.size() &&
	equal(costs.begin(), costs.end(), other.costs.begin()) &&
	other.rhs.size() == other.rhs.size() &&
	equal(rhs.begin(), rhs.end(), other.rhs.begin()) &&
	other.matbeg.size() == other.matbeg.size() &&
	equal(matbeg.begin(), matbeg.end(), other.matbeg.begin()) &&
	other.matcnt.size() == other.matcnt.size() &&
	equal(matcnt.begin(), matcnt.end(), other.matcnt.begin()) &&
	other.matind.size() == other.matind.size() &&
	equal(matind.begin(), matind.end(), other.matind.begin()) &&
	other.matval.size() == other.matval.size() &&
	equal(matval.begin(), matval.end(), other.matval.begin())
    ) {
	 same = true;
    }

    return same;
}



bool IntegerLinearProgram::operator!=(const IntegerLinearProgram& other) const {
    return !(*this == other);
}



///
/// class AdaptiveEnergiesFormulation
///
  AdaptiveEnergiesFormulation::AdaptiveEnergiesFormulation( double distance_threshold, string localisation_feature, unsigned int max_nearest_neighbors ) : distance_threshold_(distance_threshold), loc_feature_(localisation_feature) {
    shared_ptr<ConstantEnergy> e(new ConstantEnergy() );
    if(max_nearest_neighbors < 2) {
      throw std::logic_error("AdaptiveEnergiesFormulation::AdaptiveEnergiesFormulation(): number of nearest neighbors has to be at least two; divisions couldn't be tracked, else");
    }
    max_nearest_neighbors_ = max_nearest_neighbors;

    xyz_to_idx_[0] = 0;
    xyz_to_idx_[1] = 1;
    xyz_to_idx_[2] = 2;

    div_energy_ = e;
    mov_energy_ = e;
    dis_energy_ = e;
    app_energy_ = e;
}

AdaptiveEnergiesFormulation::AdaptiveEnergiesFormulation(shared_ptr<const TertiaryEnergy> division,
							 shared_ptr<const BinaryEnergy> move,
							 shared_ptr<const UnaryEnergy> disappearance,
							 shared_ptr<const UnaryEnergy> appearance,
							 double distance_threshold,
							 string localisation_feature,
							 unsigned int max_nearest_neighbors,
							 double min_division_angle
	) : distance_threshold_(distance_threshold), loc_feature_(localisation_feature), min_division_angle_(min_division_angle) {
    if(max_nearest_neighbors < 2) {
      throw std::logic_error("AdaptiveEnergiesFormulation::AdaptiveEnergiesFormulation(): number of nearest neighbors has to be at least two; divisions couldn't be tracked, else");
    }
    if(min_division_angle < 0) {
        throw std::logic_error("AdaptiveEnergiesFormulation::AdaptiveEnergiesFormulation(): minimal division angle has to be positive");
    }
    max_nearest_neighbors_ = max_nearest_neighbors;

    xyz_to_idx_[0] = 0;
    xyz_to_idx_[1] = 1;
    xyz_to_idx_[2] = 2;
    
    div_energy_ = division;
    mov_energy_ = move;
    dis_energy_ = disappearance;
    app_energy_ = appearance;    
}



pair<shared_ptr<AdaptiveEnergiesIlp>, vector<Event> > 
AdaptiveEnergiesFormulation::formulate_ilp(const Traxels& prev, const Traxels& curr) const {
    int DIM = 3;

    map<unsigned int, size_t> enumr_prev = enumerate_traxels( prev );
    map<unsigned int, size_t> enumr_curr = enumerate_traxels( curr );

    shared_ptr<AdaptiveEnergiesIlp> ilp(new AdaptiveEnergiesIlp( enumr_prev.size(), enumr_curr.size() ));
    NearestNeighborSearch nn( curr, loc_feature_, DIM );
    vector<unsigned int> dim_to_idx(xyz_to_idx_, xyz_to_idx_ + 3);
    nn.Dim_to_idx( dim_to_idx );
    vector<Event> events;

    //
    // loop over all previous traxels and generate all possible move and division events originating from them
    //
    for( Traxels::const_iterator key_value_pair = prev.begin(); key_value_pair != prev.end(); ++key_value_pair ){
	Traxel traxel = key_value_pair->second;

	// search for neighboring subsequent traxels
	unsigned int count = nn.count_in_range( traxel, distance_threshold_);
	// cap number of nearest neighbors
	if(count > max_nearest_neighbors_) {
	  count = max_nearest_neighbors_;
	}
	map<unsigned int, double> neighbors = nn.knn_in_range( traxel, distance_threshold_, count );
	assert(count == neighbors.size() );

	// add a column to the constraint matrix for every potential move
	for( map<unsigned int, double>::iterator id_dist2 = neighbors.begin(); id_dist2 != neighbors.end(); ++id_dist2 ) {
	    Traxel to = curr.find(id_dist2->first)->second;
	    double cost = (*mov_energy_)(traxel, to, prev, curr)
		-(*dis_energy_)(traxel, prev, curr)
		-(*app_energy_)(to, prev, curr);

	    ilp->add_move( enumr_prev[traxel.Id], enumr_curr[id_dist2->first], cost );
	    
	    Event e;
	    e.type = Event::Move;
	    e.energy = cost;
	    e.traxel_ids.push_back(traxel.Id);
	    e.traxel_ids.push_back(id_dist2->first);
	    events.push_back( e );
	}

	// add a column to the constraint matrix for every potential division
	for( map<unsigned int, double>::const_iterator id_dist2 = neighbors.begin(); id_dist2 != neighbors.end(); ++id_dist2 ) {
	    map<unsigned int, double>::const_iterator look_ahead = id_dist2;
	    advance(look_ahead, 1);
	    if( look_ahead == neighbors.end() ) {
		continue;
	    }
	    for( map<unsigned int, double>::const_iterator id_dist2_other = look_ahead; id_dist2_other!= neighbors.end(); ++ id_dist2_other) {
		Traxel to1 = curr.find(id_dist2->first)->second;
		Traxel to2 = curr.find(id_dist2_other->first)->second;

		// only add division with minimal angle
		double angle = -1;
		if( loc_feature_ == "com" ) {
		  traxel.set_locator(new ComLocator());
		    angle = traxel.angle( to1, to2);
		} else if ( loc_feature_ == "maxintpos" ) {
		  traxel.set_locator(new IntmaxposLocator);
		    angle = traxel.angle( to1, to2);
		} else {
		    throw std::runtime_error("AdaptiveEnergiesFormualtion::operator(): unsupported localisation feature");
		}
		if( angle >= min_division_angle_ ) {
		    double cost = (*div_energy_)(traxel, to1, to2, prev, curr)
		    -(*dis_energy_)(traxel, prev, curr)
		    -(*app_energy_)(to1, prev, curr)
		    -(*app_energy_)(to2, prev, curr);

		    ilp->add_division( enumr_prev[traxel.Id], enumr_curr[id_dist2->first], enumr_curr[id_dist2_other->first], cost );

		    Event e;
		    e.type = Event::Division;
		    e.energy = cost;
		    e.traxel_ids.push_back(traxel.Id);
		    e.traxel_ids.push_back(id_dist2->first);
		    e.traxel_ids.push_back(id_dist2_other->first);
		    events.push_back( e );
		}
	    }
	}
    }

    return pair<shared_ptr<AdaptiveEnergiesIlp>, vector<Event> >(ilp, events);
}



shared_ptr<const TertiaryEnergy> AdaptiveEnergiesFormulation::Division_energy() const {
    return div_energy_;
}
AdaptiveEnergiesFormulation&  AdaptiveEnergiesFormulation::Division_energy(shared_ptr<const TertiaryEnergy> newEnergy ) {
    div_energy_ = newEnergy;
    return *this;
}

shared_ptr<const BinaryEnergy>  AdaptiveEnergiesFormulation::Move_energy() const {
    return mov_energy_;
}
AdaptiveEnergiesFormulation&  AdaptiveEnergiesFormulation::Move_energy(shared_ptr<const BinaryEnergy> newEnergy ) {
    mov_energy_ = newEnergy;
    return *this;
}

shared_ptr<const UnaryEnergy>  AdaptiveEnergiesFormulation::Disappearance_energy() const {
    return dis_energy_;
}
AdaptiveEnergiesFormulation&  AdaptiveEnergiesFormulation::Disappearance_energy(shared_ptr<const UnaryEnergy> newEnergy ) {
    dis_energy_ = newEnergy;
    return *this;
}

shared_ptr<const UnaryEnergy>  AdaptiveEnergiesFormulation::Appearance_energy() const {
    return app_energy_;
}
AdaptiveEnergiesFormulation&  AdaptiveEnergiesFormulation::Appearance_energy(shared_ptr<const UnaryEnergy> newEnergy ) {
    app_energy_ = newEnergy;
    return *this;
}

double AdaptiveEnergiesFormulation::Distance_threshold() const {
    return distance_threshold_;
}
AdaptiveEnergiesFormulation&  AdaptiveEnergiesFormulation::Distance_threshold( double threshold ) {
    distance_threshold_ = threshold;
    return *this;
}

AdaptiveEnergiesFormulation& AdaptiveEnergiesFormulation::Loc_feature(std::string loc_feature) {
    loc_feature_ = loc_feature;
    return *this;
}
std::string AdaptiveEnergiesFormulation::Loc_feature() const {
    return loc_feature_;
}

AdaptiveEnergiesFormulation& AdaptiveEnergiesFormulation::Xyz_to_idx( const unsigned int*const xyz_to_idx ) {
    xyz_to_idx_[0] = xyz_to_idx[0];
    xyz_to_idx_[1] = xyz_to_idx[1];
    xyz_to_idx_[2] = xyz_to_idx[2];
    return *this;
}
const unsigned int* AdaptiveEnergiesFormulation::Xyz_to_idx() const {
    return xyz_to_idx_;
}



map<unsigned int, size_t> AdaptiveEnergiesFormulation::enumerate_traxels(const Traxels& traxels) const {
    map<unsigned int, size_t> enumeration;
    size_t count = 0;
    for(Traxels::const_iterator traxel = traxels.begin(); traxel != traxels.end(); ++traxel) {
	enumeration[traxel->second.Id] = count;
	++count;
    }

    return enumeration;
}



///
/// class AdaptiveEnergiesIlp
///
AdaptiveEnergiesIlp::AdaptiveEnergiesIlp( int n_total_from, int n_total_to ) 
    : n_total_from_(n_total_from), n_total_to_(n_total_to) {
    rhs = vector<double>(n_total_to + n_total_from, 1.);
    nConstr = n_total_from + n_total_to;
    }

AdaptiveEnergiesIlp& 
AdaptiveEnergiesIlp::add_move(int from_idx, int to_idx, double cost) {
    nVars += 1;
    nNonZero += 2;
    
    costs.push_back( cost );
    matbeg.push_back( matval.size() );
    matcnt.push_back( 2 );
    
    matind.push_back( from_idx );
    matind.push_back( n_total_from_ + to_idx );
    
    matval.push_back( 1. );
    matval.push_back( 1. );

    assert( this->is_consistent() );
    return *this;
}

AdaptiveEnergiesIlp& 
AdaptiveEnergiesIlp::add_division(int from_idx, int to_idx1, int to_idx2, double cost) {
    nVars += 1;
    nNonZero += 3;
    
    costs.push_back( cost );
    matbeg.push_back( matval.size() );
    matcnt.push_back( 3 );
    
    matind.push_back( from_idx );
    matind.push_back( n_total_from_ + to_idx1 );
    matind.push_back( n_total_from_ + to_idx2 );
    
    matval.push_back( 1. );
    matval.push_back( 1. );
    matval.push_back( 1. );

    assert( is_consistent() );

    return *this;
}



////
//// NearestNeighborSearch
////
NearestNeighborSearch::NearestNeighborSearch(const Traxels& traxels, const string& localisation_feature, int dim) : 
  loc_feature_(localisation_feature), dim_(dim), points_(NULL) {
  dim_to_idx_ = vector<unsigned int>(dim_, 0);
  for(size_t i = 0; i < dim_to_idx_.size(); ++i) {
    dim_to_idx_[i] = i;
  }

  if(!traxels.empty()) {
    this->define_point_set( traxels );
    try {
	kd_tree_ = boost::shared_ptr<ANNkd_tree>( new ANNkd_tree( points_, traxels.size(), dim_ ) );
    } catch(...) {
	annDeallocPts(points_);
	points_ = NULL;
	throw;
    }
  }
}

NearestNeighborSearch::~NearestNeighborSearch() {
	if(points_ != NULL) {
	    annDeallocPts(points_);
	    points_ = NULL;
	}
}



map<unsigned int, double> NearestNeighborSearch::knn_in_range( const Traxel& query, double radius, unsigned int knn ) {
    if( radius < 0 ) {
	throw "knn_in_range: radius has to be non-negative.";
    }

    map<unsigned int, double> return_value;

    // empty search space?
    if(points_ == NULL && kd_tree_.get() == NULL) {
      return return_value;
    }

    // allocate
    ANNpoint query_point( NULL );
    query_point = this->point_from_traxel(query);
    if( query_point == NULL ) {
	throw "query point allocation failure";
    }

    // search
    try {
	scoped_array<ANNidx> nn_indices( new ANNidx[knn] );
	scoped_array<ANNdist> nn_distances( new ANNdist[knn] );

	const int points_in_range = kd_tree_->annkFRSearch( query_point, radius*radius, knn,
                                         nn_indices.get(), nn_distances.get());

	if( points_in_range < 0 ) {
	    throw "knn_in_range: ANN search return negative number of nearest neighbors";
	}

	// construct return value
	
	// there may be less points in range, than nearest neighbors demanded
	const int actual = (static_cast<int>(knn) < points_in_range) ? knn : points_in_range;
	for( ANNidx i = 0; i < actual; ++i) {
	    return_value[ point_idx2traxel_id_[nn_indices[i]] ] = nn_distances[i];
	}
    } catch(...) {
	if( query_point != NULL) {
	    annDeallocPt( query_point );
	}
	throw;
    }

    // clean up
    if( query_point != NULL) {
	annDeallocPt( query_point );
    }

    return return_value;
}



unsigned int NearestNeighborSearch::count_in_range( const Traxel& query, double radius ) {
    if( radius < 0 ) {
	throw "count_in_range: radius has to be non-negative.";
    }

    // empty search space?
    if(points_ == NULL && kd_tree_.get() == NULL) {
      return 0;
    }

    // allocate
    ANNpoint query_point( NULL );
    query_point = this->point_from_traxel(query);
    if( query_point == NULL ) {
	throw "query point allocation failure";
    }

    // search
    int points_in_range;
    try {
	// search with 0 nearest neighbors -> returns just a range count
	points_in_range = kd_tree_->annkFRSearch( query_point, radius*radius, 0 );

	if( points_in_range < 0 ) {
	    throw "knn_in_range: ANN search return negative number of nearest neighbors";
	}
    } catch(...) {
	if( query_point != NULL) {
	    annDeallocPt( query_point );
	}
	throw;
    }

    // clean up
    if( query_point != NULL) {
	annDeallocPt( query_point );
    }

    return points_in_range;
}



NearestNeighborSearch& NearestNeighborSearch::Dim_to_idx( std::vector<unsigned int> dim_to_idx ) {
    dim_to_idx_ = dim_to_idx;
    return *this;
}

std::vector<unsigned int> NearestNeighborSearch::Dim_to_idx() const {
    return dim_to_idx_;
}



void NearestNeighborSearch::define_point_set( const Traxels& traxels ) {
    // allocate memory for kd-tree nodes
    size_t traxel_number = traxels.size(); 
    points_ = annAllocPts( traxel_number, dim_ );
    if( points_ == NULL ){
      throw "Allocation of points for kd-tree failed";
    }

    // fill the nodes with coordinates
    try {
	point_idx2traxel_id_.clear();
	size_t i = 0;
	for( Traxels::const_iterator traxel = traxels.begin(); traxel != traxels.end(); ++traxel, ++i) {
	  ANNpoint point = points_[i];
	  for( int j=0; j<dim_; ++j ){
	      FeatureMap fm = traxel->second.features;
	      point[j] = fm[loc_feature_][dim_to_idx_[j]];
	  }
	  // save point <-> traxel association
	  point_idx2traxel_id_[i] = traxel->second.Id;
	}
    } catch(...) {
	annDeallocPts(points_);
	points_ = NULL;
	point_idx2traxel_id_.clear();
	throw;
    }
}



ANNpoint NearestNeighborSearch::point_from_traxel( const Traxel& traxel ) {
    ANNpoint point = annAllocPt( dim_ );
    for( int i=0; i<dim_; ++i ){
	FeatureMap fm = traxel.features;
	point[i] = fm[loc_feature_][dim_to_idx_[i]];
    }
    return point;
}


} /* namespace Tracking */
