#ifndef LP_SOLVE_SOLVER_H
#define LP_SOLVE_SOLVER_H

#include "ilp_solver.h"

#ifndef USE_CPLEX

namespace Tracking {

class LpSolveSolver : public IlpSolver {
public:  
  virtual int solve(  int nVars, int nConstr, int nNonZero,
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
};

}

#endif /* USE_CPLEX */

#endif /* LP_SOLVE_SOLVER_H */


