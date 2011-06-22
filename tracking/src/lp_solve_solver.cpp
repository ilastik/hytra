#include "lp_solve_solver.h"

#ifndef USE_CPLEX

#include <iostream>
#include <sstream>
#include "lp_lib.h"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using boost::shared_array;
using std::ostringstream;

namespace Tracking {


string LpSolveSolver::getName() const {
  return "LP_SOLVE 5.5";
}

// Solve an integer linear program using the LP_SOLVE solver
int LpSolveSolver::solve( int nVars, int nConstr, int nNonZero,
                          const vector<double>& costs,
                          const vector<double>& rhs,
                          const vector<int>& matbeg,
                          const vector<int>& matcnt,
                          const vector<int>& matind,
                          const vector<double>& matval,
                          const string& problemString,
                          double& finalCost,
                          shared_array<double>& finalVars,
                          int& solverOutput,
                          string& outputString ){
  int output = 1;
  lprec* lp = NULL;
  try {
    lp = make_lp(nConstr, nVars);
    if( lp == NULL ){
      throw "LP problem " + problemString + " cannot be initialized"; 
    }
    // fill the constraint matrix by columns
    for( int c=0; c<nVars; ++c ){
      int nValues = matcnt[c]+1;
      REAL *colVals = new REAL[nValues];
      int *rowNrs = new int[nValues];
      colVals[0] = costs[c];
      rowNrs[0] = 0;
      for( int v=1; v<nValues; ++v ){
        colVals[v] = matval[matbeg[c]+v-1];
        rowNrs[v] = matind[matbeg[c]+v-1]+1;
      }
      unsigned char success = set_columnex(lp, c+1, nValues, colVals, rowNrs);
      delete[] colVals;
      delete[] rowNrs;
      if( !success ){
        ostringstream errMsg;
        errMsg << "Error while setting column no. " << c  << " in LP problem " 
                  + problemString;
        throw errMsg.str().c_str();
      }
      success = set_binary( lp, c+1, TRUE );
      if( !success ){
        ostringstream errMsg;
        errMsg << "Error while setting column no. " << c << " in " + problemString +
                  " to be binary";
        throw errMsg.str();
      }
    }
    // set the right-hand side vector of the constraints
    shared_array<REAL> extendedRhs( new REAL[nConstr+1] );
    extendedRhs[0] = 0.;
    for( int r=1; r<=nConstr; ++r ){
      extendedRhs[r] = rhs[r-1];
      unsigned char success = set_constr_type( lp, r, LE );
      if( !success ){
        ostringstream out;
        out << "Error while setting constraint type in " + problemString 
               + " for row no. " << r;
        throw out.str();
      }
    }
    set_rh_vec(lp, extendedRhs.get() );
    set_minim( lp );
    string solveString;
    solverOutput = ::solve( lp );
    finalCost = get_working_objective( lp );
    finalVars.reset( new double[nVars] );
    unsigned char success = get_variables( lp, finalVars.get() );
    if( !success ){
      throw "Error while extracting the final variables from " + problemString;
    }
    switch( solverOutput ){
    case NOMEMORY: outputString = "Out of memory while solving LP program"; break;
    case OPTIMAL: outputString = "Proper convergence"; output = 0; break;
    case SUBOPTIMAL: outputString = "Suboptimal solution found"; output = 0; break;
    case INFEASIBLE: outputString =  "Model is infeasible"; break;
    case UNBOUNDED: outputString =  "Model is unbounded"; break;
    case DEGENERATE: outputString =  "Model is degenerate"; break;
    case NUMFAILURE: outputString =  "Numerical failure during LP"; break;
    case USERABORT: outputString =  "LP aborted by user"; break;
    case TIMEOUT: outputString =  "Aborted because of time-out"; break;
    case PRESOLVED: outputString =  "Model can be presolved perfectly"; break;
    case PROCFAIL: outputString =  "Branch and bound failed"; break;
    case PROCBREAK: outputString =  "Break at first / break at value"; break;
    case FEASFOUND: outputString =  "Feasible branch and bound solution found";
                    break;
    case NOFEASFOUND: outputString =  "No feasible branch and bound solution found"; 
                      break;
    default: outputString = "Unknown output code - consult LP_SOLVE manual"; break;
    }
  } catch ( string& str ){
    cerr << str << endl;
    output = 1;
  }
  if( lp ){
    delete_lp( lp );
  }
  return output;
}

}
#endif /* USE_CPLEX */
