#ifndef ILP_CONSTRUCTION_H
#define ILP_CONSTRUCTION_H

#include <utility>
#include <vector>
#include <ANN/ANN.h>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>

#include "energy.h"
#include "event.h"
#include "traxels.h"

namespace Tracking {
    /**
     * The Formulation of an Integer Linear Program.
     *
     * suitable to be fed into an IlpSolve
     */
    struct IntegerLinearProgram {
      // We assume that all variables must be binary, and that all
      // inequalities are of the <= form.
      //
      // Constraint matrix is stored in matval in column-major order.
      //
      // nVars - number of variables (columns of constraint matrix)
      // nConstr - number of constraints (rows of constraint matrix)
      // nNonZero - number of non-zero constraint matrix entries
      // costs - costs of setting a variable to 1 (length nVars)
      // rhs - right-hand sides of inequalities (length nConstr)
      // matbeg - index of the beginning of every column in the 
      //          coefficient array matval (ascending, length nVars)
      // matcnt - number of nonzero elements in each column
      // matind - row numbers of the coefficients in matval (length nNonZero)
      // matval - array of nonzero coefficients (length nNonZero)
      //
      //
      // Example:
      //
      // min.: 1*x_0->0 + 2*x_0->1 + 3* x_0->0+1
      //
      // s.t.:
      //
      // | 1 1 1 |   | x_0->0   |    | 1 |
      // | 1 0 1 | x | x_0->1   | <= | 1 |
      // | 0 1 1 |   | x_0->0+1 |    | 1 |
      //
      // nVars:    3
      // nConstr:  3
      // nNonZero: 7
      // costs: 1,2,3
      // rhs: 1,1,1
      // matbeg: 0,2,4
      // matcnt: 2,2,3
      // matind: 0,1,0,2,0,1,2
      // matval: 1,1,1,1,1,1,1
      //
      IntegerLinearProgram() : nVars(0), nConstr(0), nNonZero(0) {};

      int nVars;
      int nConstr;
      int nNonZero;
      std::vector<double> costs;
      std::vector<double> rhs;
      std::vector<int> matbeg;
      std::vector<int> matcnt;
      std::vector<int> matind;
      std::vector<double> matval;

      bool is_consistent() const;
      bool operator==(const IntegerLinearProgram& other) const;
      bool operator!=(const IntegerLinearProgram& other) const;
    };



    class AdaptiveEnergiesIlp : public IntegerLinearProgram {
	public:
	AdaptiveEnergiesIlp( int n_total_from , int n_total_to );

	AdaptiveEnergiesIlp& add_move(int from_idx, int to_idx, double cost);
	AdaptiveEnergiesIlp& add_division(int from_idx, int to_idx1, int to_idx2, double cost);

	private:
	const int n_total_from_;
	const int n_total_to_;
    };



    class AdaptiveEnergiesFormulation {
    public:
      /** Construct with default ConstantEnergy instances. */
      AdaptiveEnergiesFormulation( double distance_threshold = 50,
				   std::string localisation_feature = "com", 
				   unsigned int max_nearest_neighbors=6);
      
      AdaptiveEnergiesFormulation(boost::shared_ptr<const TertiaryEnergy> division,
				  boost::shared_ptr<const BinaryEnergy> move,
				  boost::shared_ptr<const UnaryEnergy> disappearance,
				  boost::shared_ptr<const UnaryEnergy> appearance,
				  double distance_threshold = 50,
				  std::string localisation_feature = "com",
				  unsigned int max_nearest_neighbors=6,
				  double min_division_angle = 0
				  );
      
        std::pair<boost::shared_ptr<AdaptiveEnergiesIlp>, std::vector<Event> > 
	formulate_ilp(const Traxels& prev, const Traxels& curr) const;

	// setter / getter
	boost::shared_ptr<const TertiaryEnergy> Division_energy() const;
	AdaptiveEnergiesFormulation& Division_energy(boost::shared_ptr<const TertiaryEnergy> newEnergy );
	boost::shared_ptr<const BinaryEnergy> Move_energy() const;
	AdaptiveEnergiesFormulation& Move_energy(boost::shared_ptr <const BinaryEnergy> newEnergy );
	boost::shared_ptr<const UnaryEnergy> Disappearance_energy() const;
	AdaptiveEnergiesFormulation& Disappearance_energy(boost::shared_ptr<const UnaryEnergy> newEnergy );
	boost::shared_ptr<const UnaryEnergy> Appearance_energy() const;
	AdaptiveEnergiesFormulation& Appearance_energy(boost::shared_ptr<const UnaryEnergy> newEnergy );

	AdaptiveEnergiesFormulation& Distance_threshold(double threshold);
	double Distance_threshold() const;

	AdaptiveEnergiesFormulation& Loc_feature(std::string loc_feature);
	std::string Loc_feature() const;

	AdaptiveEnergiesFormulation& Xyz_to_idx( const unsigned int*const xyz_to_idx );
	const unsigned int* Xyz_to_idx() const;

	protected:
	std::map<unsigned int, size_t> enumerate_traxels(const Traxels& traxels) const;

	private:
	boost::shared_ptr<const TertiaryEnergy> div_energy_;
	boost::shared_ptr<const BinaryEnergy> mov_energy_;
	boost::shared_ptr<const UnaryEnergy> dis_energy_;
	boost::shared_ptr<const UnaryEnergy> app_energy_;
	
	double distance_threshold_;
	std::string loc_feature_;
	unsigned int max_nearest_neighbors_;
	unsigned int xyz_to_idx_[3];
	double min_division_angle_;
    };


    namespace legacy {
    class NearestNeighborSearch {
	public:
	    NearestNeighborSearch( const Traxels& traxels, 
				   const std::string& localisation_feature = "com", 
				   int dim = 3);
	    ~NearestNeighborSearch();
	
	     /**
	      * Returns (traxel id, distance*distance) map.
	      */
	    std::map<unsigned int, double> knn_in_range( const Traxel& query, double radius, unsigned int knn );
	    unsigned int count_in_range( const Traxel& query, double radius );
    
	    // dim_to_idx translates dimensions to indices in the feature array.
	    // default for n dimensions is just the range 0:n
	    NearestNeighborSearch& Dim_to_idx( std::vector<unsigned int> dim_to_idx );
	    std::vector<unsigned int> Dim_to_idx() const;

	private:
	    /**
	     * Ctor helper: define points and association between traxels and points
	     */
	    void define_point_set( const Traxels& traxels );
	    ANNpoint point_from_traxel( const Traxel& traxel );

	    std::map<unsigned int, unsigned int> point_idx2traxel_id_;
	
	    const std::string loc_feature_;
	    const int dim_;
	    std::vector<unsigned int> dim_to_idx_;
	
	    ANNpointArray points_;
	    boost::shared_ptr<ANNkd_tree> kd_tree_;
    };
    } /* namespace legacy */
}

#endif /* ILP_CONSTRUCTION_H */





