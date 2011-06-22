#ifndef CPLEX_SOLVER_H
#define CPLEX_SOLVER_H

#include "ilp_solver.h"

#ifdef USE_CPLEX

#include <ilcplex/cplex.h>

namespace Tracking {

class CplexSolver : public IlpSolver {
public:
   CplexSolver();
   ~CplexSolver();
   virtual int solve( int nVars, int nConstr, int nNonZero,
                      const std::vector<double>& costs,
                      const std::vector<double>& rhs,
                      const std::vector<int>& matbeg,
                      const std::vector<int>& matcnt,
                      const std::vector<int>& matind,
                      const std::vector<double>& matval,
                      const std::string& problemString,
                      double& finalCost,
                      boost::shared_array<double>& finalVars,
                      int& solverOutput,
                      std::string& outputString );
   virtual std::string getName() const;
private:
  CPXENVptr env_;
  // report a CPLEX error message to stderr
  void reportCplexError( int errcode );
  // deactivate copying
  CplexSolver(const CplexSolver&);
  CplexSolver& operator=(const CplexSolver&);
};

}

#endif /* USE_CPLEX */

#endif /* CPLEX_SOLVER_H */
