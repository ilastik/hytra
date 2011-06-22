#ifndef __CT_GVF_GENERATOR__
#define __CT_GVF_GENERATOR__

#include <iostream>
#include <vector>
#include <vigra/multi_array.hxx>
#include "vigraAlgorithmPackage.hxx"
#include <vigra/multi_distance.hxx>
#include <ctime>
#include <math.h>

using namespace vigra;

class ctGVFGenerator
{
public:
    // constructor
    ctGVFGenerator(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctGVFGenerator: ";
        
        sigma = 1.2;
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cerr << prefix << "parameters ->" << std::endl;
        std::cerr << "\t\t\t\t sigma = " << sigma << std::endl;
    }
    
	// operator
    template<int DIM, class INT, class FLOAT >
	void run(MultiArray<DIM, INT > &seeds, MultiArray<DIM+1, FLOAT > &gvf)
    {
        typedef typename MultiArray<2, FLOAT >::difference_type Shape2D;
        typedef typename MultiArray<DIM, FLOAT >::difference_type Shape;
        typedef typename MultiArray<DIM+1, FLOAT >::difference_type ShapeGVF;
        
        if (verbose)
            std::cerr << prefix << "compute the centroid of individual seeds " << std::endl;
        // get the centers of the connect components
        FindMinMax<INT > minmax;
        inspectMultiArray(srcMultiArrayRange(seeds), minmax);
        MultiArray<2, FLOAT > centers;
        centers.reshape(Shape2D(minmax.max, DIM), 0);
        MultiArray<2, FLOAT > sizes;
        sizes.reshape(Shape2D(minmax.max, DIM), 0);
        for (int x=0; x<seeds.shape(0); x++) {
            for (int y=0; y<seeds.shape(1); y++) {
                for (int z=0; z<seeds.shape(2); z++) {
                    int id = seeds(x, y, z);
                    if (id == 0)
                        continue;
                    
                    centers(id-1, 0) += x; 
                    centers(id-1, 1) += y; 
                    centers(id-1, 2) += z; 
                    sizes(id-1, 0) += 1;
                    sizes(id-1, 1) += 1;
                    sizes(id-1, 2) += 1;
                }
            }
        }
        centers /= sizes;
        
        // perform the distance transform
        if (verbose)
            std::cerr << prefix << "perform the distance transform " << std::endl;
        MultiArray<DIM, FLOAT > bw;
        bw.reshape(seeds.shape(), 0);
        MultiArray<DIM, FLOAT > dist;
        dist.reshape(seeds.shape(), 0);
        for (int i=0; i<rowCount(centers); i++) 
            bw(floor(centers(i, 0)), floor(centers(i, 1)), floor(centers(i, 2))) = 1;
        separableMultiDistSquared(srcMultiArrayRange(bw), destMultiArray(dist), true);
        
        // compute the gradient
        if (verbose)
            std::cerr << prefix << "compute the gradient " << std::endl;
        vigraGaussianGradient3<FLOAT >(dist, sigma, gvf);
        
        // normalize
        for (int x=0; x<seeds.shape(0); x++) {
            for (int y=0; y<seeds.shape(1); y++) {
                for (int z=0; z<seeds.shape(2); z++) {
                    FLOAT mag = sqrt(gvf(x, y, z, 0)*gvf(x, y, z, 0) + 
                            gvf(x, y, z, 1)*gvf(x, y, z, 1) + 
                            gvf(x, y, z, 2)*gvf(x, y, z, 2));
                    gvf(x, y, z, 0) = gvf(x, y, z, 0)/mag;
                    gvf(x, y, z, 1) = gvf(x, y, z, 1)/mag;
                    gvf(x, y, z, 2) = gvf(x, y, z, 2)/mag;
                }
            }
        }
    }

private:
    std::string prefix;
    int verbose;
    float sigma;
};

#endif /* __CT_GVF_GENERATOR__ */