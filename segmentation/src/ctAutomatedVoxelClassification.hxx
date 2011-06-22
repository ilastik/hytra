#ifndef __CT_AUTOMATED_VOXEL_CLASSIFICATION__
#define __CT_AUTOMATED_VOXEL_CLASSIFICATION__

#include <iostream>
#include <vector>
#include <vigra/multi_array.hxx>
#include <vigra/multi_morphology.hxx>
#include "ctFeatureExtractor.hxx"
#include "ctClassifierTrainer.hxx"
#include "ctProbabilityMapPredictor.hxx"
#include <ctime>

using namespace vigra;

class ctAutomatedVoxelClassification
{
public:
    // constructor: load parameters from the ini file
    ctAutomatedVoxelClassification(const CSimpleIniA &ini, int verbose = false) 
        : verbose(verbose), fe(ini, verbose), ct(ini, verbose), pmp(ini, verbose)
    {
        prefix = "ctAutomatedVoxelClassification: ";
        
        minIntensity = atof(ini.GetValue(INI_SECTION_PREPROCESSING, "intensity_threshold", "0"));
        nTrees = atoi(ini.GetValue(INI_SECTION_RANDOM_FOREST, "n_trees", "255"));

        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
        
    // print parameters
    void print()
    {
        std::cout << prefix << "parameters ->" << std::endl;
        std::cout << "\t\t\t\t minIntensity = " << minIntensity << std::endl;     
    }
    
	// operator
    template<int DIM, class INT, class FLOAT >
	void run(MultiArrayView<DIM, FLOAT > data, 
            MultiArrayView<DIM, INT > seeds, 
            MultiArrayView<DIM, INT > watersheds, 
            MultiArray<DIM, INT > &mask,
            MultiArray<DIM, FLOAT > &probmap)
    {
        // timing
        clock_t start;
        
        // formulate labels: 1 - fg; 2 - bg
        //MultiArray<DIM, INT > watersheds_(watersheds);
        //multiBinaryDilation(srcMultiArrayRange(watersheds_), destMultiArray(watersheds_), 1);
        MultiArray<DIM, FLOAT > labels(seeds);      // random forest input, same data type as features
        std::replace_if(labels.begin(), labels.end(), std::bind2nd(std::not_equal_to<INT >(), 0), 3);
        labels += watersheds;
        std::replace_if(labels.begin(), labels.end(), std::bind2nd(std::equal_to<INT >(), 1), 2);
        std::replace_if(labels.begin(), labels.end(), std::bind2nd(std::equal_to<INT >(), 3), 1);
        
        // extract features
        start = clock();
        if (verbose)
            std::cerr << prefix << "start feature extraction " << std::endl;
        MultiArray<DIM+1, FLOAT > features;
        fe.run<DIM, FLOAT >(data, features);
        if (verbose) {
            std::cerr << prefix << "feature object size = " << features.shape() << std::endl;
            std::cerr << prefix << "time for feature extraction = " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
        }
        
        // train the classifier
        start = clock();
        vigra::RandomForestOptions options;
        options.tree_count(nTrees);
        RandomForest<FLOAT > classifier(options);
        
        double oobError = ct.run<DIM, FLOAT >(features, labels, classifier);
        if (verbose) {
            std::cerr << prefix << "random forest oob error = " << oobError << std::endl;
            std::cerr << prefix << "time for classifier training = " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
        }
        
        // create the mask
        mask.reshape(labels.shape());
        std::replace_copy_if(labels.begin(), labels.end(), mask.begin(), std::bind2nd(std::equal_to<INT >(), 2), 0);
//        mask.copy(labels);
//        std::replace_if(mask.begin(), mask.end(), std::bind2nd(std::equal_to<INT >(), 2), 0);
        for (int i=0; i<mask.elementCount(); i++) {
            if (data[i] < minIntensity)
                mask[i] = 2;
        }
        if (verbose) {
            std::cerr << prefix << "Percentage of fixed foreground: " << 
                    std::count_if(mask.begin(), mask.end(), std::bind2nd(std::equal_to<FLOAT >(), 1)) / float(mask.elementCount()) 
                    << std::endl;
            std::cerr << prefix << "Percentage of fixed background: " << 
                    std::count_if(mask.begin(), mask.end(), std::bind2nd(std::equal_to<FLOAT >(), 2)) / float(mask.elementCount())
                    << std::endl;
        }
        
        // perform the prediction
        start = clock();
        pmp.run<DIM, INT, FLOAT >(classifier, features, probmap, mask);
        if (verbose)
            std::cerr << prefix << "time for prediction " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
    }

private:
    int verbose;
    std::string prefix;
    ctFeatureExtractor fe;
    ctClassifierTrainer ct;
    int nTrees;
    ctProbabilityMapPredictor pmp;
    float minIntensity;
};

#endif /* __CT_AUTOMATED_VOXEL_CLASSIFICATION__ */