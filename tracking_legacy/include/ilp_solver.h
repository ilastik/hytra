#ifndef ILP_SOLVER_H
#define ILP_SOLVER_H

#include <string>
#include <vector>
#include <boost/shared_array.hpp>

namespace Tracking {

class IlpSolver {

public:
  // Common interface of the ILP solvers:
  //
  // We assume that all variables must be binary, and that all
  // inequalities are of the <= form.
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
  // finalCost - costs of the final IP solution
  // finalVars - final values of the variables at the optimum
  // solverOutput - solver-specific output flag
  // outputString - string explaining this output flag
  // 
  // The output determines whether the operation was successful (0), or
  // not (1)
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
                      std::string& outputString ) = 0;
  // Return the name string of the solver that is used
  virtual std::string getName() const = 0;
};

}

#endif /* ILP_SOLVER_H */

