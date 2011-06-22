#ifndef __CT_SIZE_FILTER__
#define __CT_SIZE_FILTER__

#include <iostream>
#include <vector>
#include <ctime>
#include "vigraAlgorithmPackage.hxx"
#include <vigra/labelvolume.hxx>
    
using namespace vigra;

class ctSizeFilter
{
public:    
    // constructor: load parameters from the ini file
    ctSizeFilter(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctSizeFilter: ";
        minSize = atoi(ini.GetValue(INI_SECTION_POST_PROCESSING, "size_threshold", "0"));
        conn = atoi(ini.GetValue(INI_SECTION_MULTISCALE_ANALYSIS, "conn", "6"));
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;       
        std::cout << "\t\t\t\t minSize = " << minSize << std::endl;
        std::cout << "\t\t\t\t conn = " << conn << std::endl;
    }
    
	// operator
    template<int DIM, class INT >
	void run(MultiArray<DIM, INT > &seeds)
    {
        typedef typename MultiArray<2, INT >::difference_type Shape2D;
        if (verbose) {
            std::cerr << prefix << "filter by size " << std::endl;
        }
        
        // get the size of the connect components
        FindMinMax<INT > minmax;
        inspectMultiArray(srcMultiArrayRange(seeds), minmax);
        MultiArray<2, INT > sizes;
        sizes.reshape(Shape2D(minmax.max, 1), 0);
        for (int i=0; i<seeds.elementCount(); i++) {
            int id = seeds[i];
            if (id == 0)
                continue;
        
            sizes(id-1, 0) += 1;
        }
        
        // filter
        MultiArray<DIM, INT > tmp;
        tmp.reshape(seeds.shape(), 0);
        for (int i=0; i<seeds.elementCount(); i++) {
            int id = seeds[i];
            if (id == 0)
                continue;
        
            tmp[i] = (sizes(id-1, 0) > minSize);
        }
        
        // rerun the cc analysis
        if (conn == 6) 
            labelVolumeWithBackground(
                    srcMultiArrayRange(tmp), 
                    destMultiArray(seeds),
                    NeighborCode3DSix(), 
                    0);
        else if (conn == 26) 
            labelVolumeWithBackground(
                    srcMultiArrayRange(tmp), 
                    destMultiArray(seeds),
                    NeighborCode3DTwentySix(), 
                    0);
    }

private:
    unsigned int minSize, conn;
    int verbose;
    std::string prefix;
};

#endif /* __CT_SIZE_FILTER__ */
