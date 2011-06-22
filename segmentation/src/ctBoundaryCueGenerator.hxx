#ifndef __CT_BOUNDARY_CUE_GENERATOR__
#define __CT_BOUNDARY_CUE_GENERATOR__

#include <iostream>
#include <vector>
#include <ctime>
#include "vigraAlgorithmPackage.hxx"
    
using namespace vigra;

class ctBoundaryCueGenerator
{
public:    
    // constructor: load parameters from the ini file
    ctBoundaryCueGenerator(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctBoundaryCueGenerator: ";
        sigma = atof(ini.GetValue(INI_SECTION_BOUNDARY_CUE, "sigma", "1.2"));
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;       
        std::cout << "\t\t\t\t sigma = " << sigma << std::endl;
    }
    
	// operator
    template<int DIM, class INT, class FLOAT >
	void run(MultiArray<DIM, INT > &watersheds, MultiArray<DIM, FLOAT > &bdcue)
    {
        if (verbose) {
            std::cerr << prefix << "compute boundary cue " << std::endl;
        }
        // reshape seg
        bdcue.reshape(watersheds.shape(), 0);
    
        // smooth with a gaussian kernel
        MultiArray<DIM, FLOAT > tmp(watersheds);
        vigraGaussianSmoothing3<FLOAT >(tmp, sigma, bdcue);
        
        // normalize
        FindMinMax<FLOAT > minmax;
        inspectMultiArray(srcMultiArrayRange(bdcue), minmax);
        MultiArray<DIM, FLOAT > matMax;
        matMax.reshape(watersheds.shape(), minmax.max);
        bdcue /= matMax;
    }

private:
    float sigma;
    int verbose;
    std::string prefix;
};

#endif /* __CT_BOUNDARY_CUE_GENERATOR__ */
