#ifndef __CT_CLASSIFIER_TRAINER__
#define __CT_CLASSIFIER_TRAINER__

#include <iostream>
#include <vector>
#include <ctime>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>
//#include "random_forest_impex.hxx"
//#include "RandomForestProgressVisitor.hxx"

#define LF_GRAYVALUE                        0x0001
#define LF_GRADIENT_MAGNITUDE               0x0002
#define LF_DIFFERENCE_OF_GAUSSIAN           0x0004
#define LF_EIGENVALUE_OF_HESSIAN            0x0008
#define LF_EIGENVALUE_OF_STRUCTURE_TENSOR   0x0010

using namespace vigra;

class ctClassifierTrainer
{
public:
    // constructor
    ctClassifierTrainer(
            double nTrees, 
            unsigned short nPosSamples, 
            unsigned short nNegSamples) : 
        nTrees(nTrees), nPosSamples(nPosSamples), nNegSamples(nNegSamples)
    {
    }
    
    // constructor: from ini file
    ctClassifierTrainer(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctClassifierTrainer: ";
        
        nTrees = atof(ini.GetValue(INI_SECTION_RANDOM_FOREST, "n_trees", "200"));
        nPosSamples = atoi(ini.GetValue(INI_SECTION_RANDOM_FOREST, "n_positive_samples", "200"));
        nNegSamples = atoi(ini.GetValue(INI_SECTION_RANDOM_FOREST, "n_negative_samples", "200"));
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
        
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;
        std::cout << "\t\t\t\t nTrees = " << nTrees << std::endl;
        std::cout << "\t\t\t\t nPosSamples = " << nPosSamples << std::endl;
        std::cout << "\t\t\t\t nNegSamples = " << nNegSamples << std::endl;
    }
    
	// operator
    template<int DIM, class FLOAT > double run(
            MultiArray<DIM + 1, FLOAT > &features, 
            MultiArray<DIM, FLOAT > &labels,
            RandomForest<FLOAT > &classifier)
    {
        typedef typename MultiArrayView<2, FLOAT >::difference_type Shape2D;
        typedef typename MultiArrayView<2, FLOAT >::difference_type ShapeFV;
        
        // set rf options
        //(classifier.set_options()).tree_count(nTrees);

        // get pos/neg sample indice
        if (verbose)
            std::cerr << prefix << "obtain pos/negative samples and randomly shuffel them " << std::endl;
        std::vector<MultiArrayIndex > indPosSamples, indNegSamples;
        for (MultiArrayIndex ind = 0; ind < labels.elementCount(); ind ++) {
            if (labels[ind] == 1)
                indPosSamples.push_back(ind);
            else if (labels[ind] == 2)
                indNegSamples.push_back(ind);
        }
        if (verbose) {
            std::cerr << prefix << "number of positive/negative samples = " << 
                    indPosSamples.size() << "/" << indNegSamples.size() << std::endl;
        }

        // random shuffel (random permutation)
        std::random_shuffle(indPosSamples.begin(), indPosSamples.end());
        std::random_shuffle(indNegSamples.begin(), indNegSamples.end());
        nPosSamples = (nPosSamples < indPosSamples.size()) ? nPosSamples : indPosSamples.size();
        nNegSamples = (nNegSamples < indNegSamples.size()) ? nNegSamples : indNegSamples.size();
        
        // create the feature vector and label vector
        if (verbose)
            std::cerr << prefix << "formulate the training feature vector and label vector " << std::endl;
        
        MultiArrayView<2, FLOAT > tmp = MultiArrayView<2, FLOAT >(
            Shape2D(labels.elementCount(), features.elementCount() / labels.elementCount()),
            features.data());
        MultiArray<2, FLOAT > vFeatures;
        vFeatures.reshape(Shape2D(nPosSamples + nNegSamples, features.shape(DIM)), 0);
        MultiArray<2, FLOAT > vLabels;
        vLabels.reshape(Shape2D(nPosSamples + nNegSamples, 1), 0);
        
        // add the positive samples
        for (MultiArrayIndex i = 0; i < nPosSamples; i ++) {
            for (MultiArrayIndex j = 0; j < features.shape(DIM); j++)
                vFeatures(i, j) = tmp(indPosSamples[i], j);
            vLabels[i] = labels[indPosSamples[i]];
        }
        
        // add the negative samples
        for (MultiArrayIndex i = 0; i < nNegSamples; i ++) {
            for (MultiArrayIndex j = 0; j < features.shape(DIM); j++)
                vFeatures(i + nPosSamples, j) = tmp(indNegSamples[i], j);
            vLabels[i + nPosSamples] = labels[indNegSamples[i]];
        }
        
        // train the classifier
        if (verbose)
            std::cerr << prefix << "start random forest training " << std::endl;
        
        classifier.learn(vFeatures, vLabels);
        
        return 0.0;
    }

private:
    double nTrees;
    unsigned short nPosSamples;
    unsigned short nNegSamples;
    std::string prefix;
    int verbose;
};

#endif /* __CT_CLASSIFIER_TRAINER__ */