#ifndef __CT_PRE_PROCESSOR__
#define __CT_PRE_PROCESSOR__

#include <iostream>
#include <vector>
#include <vigra/array_vector.hxx> 
#include <vigra/multi_array.hxx>
#include <vigra/multi_morphology.hxx>
#include "vigraAlgorithmPackage.hxx"
#include "ctIniConfiguration.hxx"
#include <vigra/labelvolume.hxx>
#include <ctime>
    
using namespace vigra;

class ctPreProcessor
{
public:
    // constructor: load parameters from the ini file
    ctPreProcessor(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctPreProcessor: ";
        getMatrix<float >(ini, INI_SECTION_PREPROCESSING, "smoothing", sigmas);
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;
        std::cout << "\t\t\t\t sigmas = " << sigmas << std::endl;
    }
    
	// operator
    template<int DIM, class INT, class FLOAT >
	void run(MultiArrayView<DIM, FLOAT > data)
    {        
        if (verbose) {
            std::cerr << prefix << "start pre-processing " << std::endl;
        }
        
        if (sigmas[0] > 0 && sigmas[1] > 0 && sigmas[2] > 0) {
            MultiArray<DIM, FLOAT > tmp(data); 
            vigraGaussianSmoothing3<FLOAT >(data, sigmas, tmp);
            data.copy(tmp);
        }
    }

private:
    Matrix<float > sigmas;
    int verbose;
    std::string prefix;
};

#endif /* __CT_PRE_PROCESSOR__ */
