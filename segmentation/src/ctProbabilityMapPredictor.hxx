#ifndef __CT_PROBABILITY_MAP_PREDICTOR__
#define __CT_PROBABILITY_MAP_PREDICTOR__

#include <iostream>
#include <vector>
#include <ctime>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

using namespace vigra;

class ctProbabilityMapPredictor
{
public:
    // constructor
    ctProbabilityMapPredictor(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctProbabilityMapPredictor: ";
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cerr << prefix << "parameters ->" << std::endl;
    }
    
	// operator
    template<int DIM, class INT, class FLOAT >
	void run(RandomForest<FLOAT > &classifier,
            MultiArray<DIM + 1, FLOAT > &features, 
            MultiArray<DIM, FLOAT > &probmap, 
            MultiArray<DIM, INT > &mask)
    {
        typedef typename MultiArray<DIM, FLOAT >::difference_type Shape;
        typedef typename MultiArray<2, FLOAT >::difference_type Shape2D;
        probmap.reshape(Shape(features.shape(0), features.shape(1), features.shape(2)), 0);
        
        MultiArrayView<2, FLOAT > vProbmap = MultiArrayView<2, FLOAT >(
            Shape2D(probmap.elementCount(), 1),
            probmap.data());
        
        MultiArrayView<2, FLOAT > vFeatures = MultiArrayView<2, FLOAT >(
            Shape2D(probmap.elementCount(), vFeatures.shape(DIM)),
            features.data());
        
        //Classify for each row.
        if (verbose) {
            std::cerr << prefix << "start random forest prediction" << std::endl;
        }
        ArrayVector<double>::const_iterator votes;
        for(int row = 0; row < rowCount(vFeatures); row ++) {
            // check the mask
            if (mask[row] == 1)         // fixed as foreground
                vProbmap[row] = 1;
            else if (mask[row] == 2)    // fixed as background
                vProbmap[row] = 0;
            else {                      // ask the classifier
                for(int k=0; k < classifier.options().tree_count_; k ++) {
                    // get weights predicted by single tree
                    votes = classifier.tree(k).predict(rowVector(vFeatures, row));
                    vProbmap[row] += votes[0];
                }
            
                vProbmap[row] /= classifier.options().tree_count_;
            }
        }
    }

private:
    std::string prefix;
    int verbose;
};

#endif /* __CT_PROBABILITY_MAP_PREDICTOR__ */