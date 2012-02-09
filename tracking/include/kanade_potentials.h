#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <stdexcept>
#include <traxels.h>
#include <cmath>

namespace Tracking {

  class KanadeIniPotential {
  public:
    KanadeIniPotential(	double temporal_threshold,
			double earliest_timestep,
			double otherwise,
			double lambda)
      : temporal_threshold_(temporal_threshold),
      earliest_timestep_(earliest_timestep),
      otherwise_(otherwise),
      lambda_(lambda) {}

    double operator()( const Traxel& tr ) {
      double dt = (tr.Timestep - earliest_timestep_);
      if( dt < temporal_threshold_) {
	return exp(-1*(dt / lambda_ ));
      } else {
	return otherwise_;
      }
    }

  private:
    double temporal_threshold_;
    double earliest_timestep_;
    double otherwise_;
    double lambda_;
  };

  class KanadeTermPotential {
  public:
    KanadeTermPotential(double temporal_threshold,
			double latest_timestep,
			double otherwise,
			double lambda)
      : temporal_threshold_(temporal_threshold),
      latest_timestep_(latest_timestep),
      otherwise_(otherwise),
      lambda_(lambda) {}

    double operator()( const Traxel& tr ) {
      double dt = (latest_timestep_ - tr.Timestep);
      if( dt < temporal_threshold_) {
	return exp(-1*(dt / lambda_ ));
      } else {
	return otherwise_;
      }
    }

  private:
    double temporal_threshold_;
    double latest_timestep_;
    double otherwise_;
    double lambda_;
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
