#include "cplex_solver.h"
#include <iostream>

#ifdef USE_CPLEX

using boost::shared_array;
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

namespace Tracking {

CplexSolver::CplexSolver() : env_(NULL) {
  int status;
  env_ = CPXopenCPLEX(&status);
  if(env_==NULL){
    reportCplexError( status );
    throw "Error while initializing CPLEX environment";
  }
}

CplexSolver::~CplexSolver(){
  int status = CPXcloseCPLEX(&env_);
  if( status ){
    cerr << "Error while releasing CPLEX environment" << endl;
  }
}

void CplexSolver::reportCplexError(int errcode){
  char errbuffer[4096];
  CPXCCHARptr outptr = CPXgeterrorstring(env_, errcode, errbuffer);
  if( outptr!=NULL ){
    cerr << errbuffer << endl;
  } else {
    cerr << "Unknown CPLEX error flag: "  << errcode << endl; 
  }
}

// Solve an integer linear program using the CPLEX solver
  int CplexSolver::solve( int nVars, int nConstr, int /*nNonZero*/,
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
                        string& outputString){
  int output = 1;
  CPXLPptr lp = NULL;
  try{
    int status = 0;
    lp = CPXcreateprob(env_, &status, problemString.c_str() );
    if( lp==NULL ){
      reportCplexError(status);
      throw "Error while creating LP problem " + problemString; 
    }
    shared_array<char> sense(new char[nConstr]);
    for( int c=0; c<nConstr; ++c ){
      sense[c] = 'L'; // <= constraint
    }
    shared_array<double> lb(new double[nVars]);
    shared_array<double> ub(new double[nVars]);
    for( int v=0; v<nVars; ++v ){
      lb[v] = 0.;
      ub[v] = 1.;
    }
    status = CPXcopylp(env_, lp, nVars, nConstr, CPX_MIN, &(costs[0]), &(rhs[0]),
                       sense.get(), &(matbeg[0]), &(matcnt[0]), &(matind[0]), 
                       &(matval[0]), lb.get(), ub.get(), NULL);
    if( status ){
      reportCplexError(status);
      throw "Error while filling ILP problem " + problemString;
    }
    shared_array<char> ctypes(new char[nVars]);
    for( int v=0; v<nVars; ++v ){
      ctypes[v] = CPX_BINARY;
    }
    status = CPXcopyctype(env_, lp, ctypes.get());
    if( status ){
      reportCplexError(status);
      throw "Error while setting variable types for problem " + problemString;
    }
    status = CPXmipopt(env_, lp);
    if( status ){
      reportCplexError(status);
      throw "Error while optimizing ILP problem " + problemString;
    }
    finalVars.reset( new double[nVars] );
    status = CPXsolution(env_, lp, &solverOutput, &finalCost, finalVars.get(), NULL,
      NULL, NULL);
    if( status ){
      reportCplexError(status);
      throw "Error while extracting the solution from the ILP problem " + problemString;
    }
    switch( solverOutput ){
    case CPXMIP_OPTIMAL: outputString = "Optimal solution found"; output = 0; break;
    case CPXMIP_OPTIMAL_INFEAS: outputString = "Optimal with unscaled infeasibilities";
      output = 0; break;
    case CPXMIP_OPTIMAL_TOL: outputString = "Optimal solution within tolerance"; 
      output = 0; break;
    case CPXMIP_TIME_LIM_FEAS: outputString = "Time limit exceeded, problem feasible";
      break;
    case CPXMIP_TIME_LIM_INFEAS: 
      outputString = "Time limit exceeded, no feasible solution"; break;
    case CPXMIP_NODE_LIM_FEAS: outputString = "Node limit exceeded, problem feasible";
      break;
    case CPXMIP_NODE_LIM_INFEAS: 
      outputString = "Node limit exceeded, no feasible solution"; break;
    case CPXMIP_UNBOUNDED: outputString = "Problem is unbounded"; break;
    case CPXMIP_SOL_LIM: outputString = "Limit on MIP solutions reached"; break;
    case CPXMIP_INFEASIBLE: outputString = "Problem is integer infeasible"; break;
    case CPXMIP_INForUNBD: outputString = "Problem infeasible or unbounded"; break;
    case CPXMIP_MEM_LIM_FEAS: outputString = "Memory limit exceeded, problem feasible";
      break;
    case CPXMIP_MEM_LIM_INFEAS: outputString = "Memory limit exceeded, no feasible "
      "solution"; break;
    case CPXMIP_ABORT_FEAS: outputString = "Aborted, problem feasible"; break;
    case CPXMIP_ABORT_INFEAS: outputString = "Aborted, no feasible solution"; break;
    case CPXMIP_FAIL_FEAS: outputString = "Failure, problem feasible"; break;
    case CPXMIP_FAIL_FEAS_NO_TREE: outputString = "Out of memory, no tree available, "
      "problem feasible"; break;
    case CPXMIP_FAIL_INFEAS: outputString = "Failure, no feasible solution"; break;
    case CPXMIP_FAIL_INFEAS_NO_TREE: outputString = "Out of memory, no tree available,"
      " no feasible solution"; break;
    default: outputString = "Unexpected error flag, consult CPLEX manual"; break;
    }
  } catch( string& str ){
    cerr << str << endl;
    output = 1;
  } 
  if( lp ){
    int out = CPXfreeprob( env_, &lp );
    if( out ){
      cerr << "Error while releasing LP problem" << endl; 
    }
  }
  return output;
}

string CplexSolver::getName() const  {
  return "CPLEX 12.1";
}

}

#endif /* USE_CPLEX */

