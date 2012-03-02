#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <stdexcept>
#include <traxels.h>
#include <cmath>
#include <sstream>
#include <ostream>

#include "field_of_view.h"

using namespace std;

namespace Tracking {

  class KanadeIniPotential {
  public:
  KanadeIniPotential( FieldOfView fov,	
		      double temporal_threshold = 15,
		      double spatial_threshold = 40,
		      double otherwise = 0.000001,
		      double lambda_temporal = 5,
		      double lambda_spatial = 30)
    : fov_(fov),
      temporal_threshold_(temporal_threshold),
      spatial_threshold_(spatial_threshold),
      otherwise_(otherwise),
      lambda_temporal_(lambda_temporal),
      lambda_spatial_(lambda_spatial)
	{}

    double operator()( const Traxel& tr ) {
      if(! fov_.contains(tr.Timestep, tr.X(), tr.Y(), tr.Z())) {
	std::stringstream ss;
	ss << "KanadeIniPotential::operator()(): traxel not contained in field of view: " << tr;
	throw std::runtime_error(ss.str());
      }
      double dt = (tr.Timestep - fov_.lower_bound()[0]);
      double ds = fov_.spatial_margin(tr.Timestep, tr.X(), tr.Y(), tr.Z());

      double pt = exp(-1*(dt / lambda_temporal_ ));
      double ps = exp(-1*(ds / lambda_spatial_ ));;

      double ret = 0;

      if( dt < temporal_threshold_ && ds < spatial_threshold_) {
	ret = pt < ps ? ps : pt;
      } else if( dt < temporal_threshold_ ) {
	ret = pt;
      } else if(ds < spatial_threshold_) {
	ret = ds;
      } else {
	ret =  otherwise_;
      }

      return ret;
    }

  private:
    FieldOfView fov_;
    double temporal_threshold_;
    double spatial_threshold_;
    double otherwise_;
    double lambda_temporal_;
    double lambda_spatial_;
  };

  class KanadeTermPotential {
  public:
  KanadeTermPotential(FieldOfView fov,	
		      double temporal_threshold = 15,
		      double spatial_threshold = 40,
		      double otherwise = 0.000001,
		      double lambda_temporal = 5,
		      double lambda_spatial = 30)
    : fov_(fov),
      temporal_threshold_(temporal_threshold),
      spatial_threshold_(spatial_threshold),
      otherwise_(otherwise),
      lambda_temporal_(lambda_temporal),
      lambda_spatial_(lambda_spatial) {}

    double operator()( const Traxel& tr ) {
      if(! fov_.contains(tr.Timestep, tr.X(), tr.Y(), tr.Z())) {
	std::stringstream ss;
	ss << "KanadeTermPotential::operator()(): traxel not contained in field of view: " << tr;
	throw std::runtime_error(ss.str());
      }
      double dt = fov_.upper_bound()[0] - tr.Timestep;
      double ds = fov_.spatial_margin(tr.Timestep, tr.X(), tr.Y(), tr.Z());

      double pt = exp(-1*(dt / lambda_temporal_ ));
      double ps = exp(-1*(ds / lambda_spatial_ ));;

      double ret = 0;

      if( dt < temporal_threshold_ && ds < spatial_threshold_) {
	ret =  pt < ps ? ps : pt;
      } else if( dt < temporal_threshold_ ) {
	ret =  pt;
      } else if(ds < spatial_threshold_) {
	ret =  ds;
      } else {
	ret =  otherwise_;
      }

      return ret;
    }

  private:
    FieldOfView fov_;
    double temporal_threshold_;
    double spatial_threshold_;
    double otherwise_;
    double lambda_temporal_;
    double lambda_spatial_;
  };

  class KanadeLinkPotential {
  public:
  KanadeLinkPotential(double lambda) : lambda_(lambda) {}

    double operator()( const Traxel& from, const Traxel& to ) {
      double d = from.distance_to(to);
      return exp(-1*(d/lambda_));
    }

  private:
    double lambda_;
  };

  class KanadeDivPotential {
  public:
  KanadeDivPotential(double lambda) : lambda_(lambda) {}
    
    double operator()( const Traxel& ancestor, const Traxel& child1, const Traxel& child2 ) {
      double d1 = ancestor.distance_to(child1);
      double d2 = ancestor.distance_to(child2);
      return exp(-1*(d1 + d2)/(2*lambda_));
    }

  private:
    double lambda_;
  };

  class KanadeFpPotential {
  public:
  KanadeFpPotential( double misdetection_rate ) : misdetection_rate_(misdetection_rate) {
      if(! (0 <= misdetection_rate && misdetection_rate <= 1 )) {
	throw std::runtime_error("KanadePotential::KanadePotential(): misdetection rate has to be between 0 and 1");
      }
    }

    double operator()( const Traxel& ) {
      return misdetection_rate_;
    }

  private:
    double misdetection_rate_;
  };

  class KanadeTpPotential {
  public:
  KanadeTpPotential( double misdetection_rate ) : fp_(misdetection_rate){
      if(! (0 <= misdetection_rate && misdetection_rate <= 1 )) {
	throw std::runtime_error("KanadePotential::KanadePotential(): misdetection rate has to be between 0 and 1");
      }
    }

  double operator()( const Traxel& t) {
    return 1 - fp_( t );
  }

  private:
    KanadeFpPotential fp_;
  };

} /* Namespace Tracking */

#endif /* POTENTIAL_H */
