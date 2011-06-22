#ifndef __CT_FEATURE_EXTRACTOR__
#define __CT_FEATURE_EXTRACTOR__

#include <iostream>
#include <vector>
#include <vigra/multi_array.hxx>
#include <vigra/multi_morphology.hxx>
#include "vigraAlgorithmPackage.hxx"
#include <ctime>

#define LF_GRAYVALUE                        0x0001
#define LF_GRADIENT_MAGNITUDE               0x0002
#define LF_DIFFERENCE_OF_GAUSSIAN           0x0004
#define LF_EIGENVALUE_OF_HESSIAN            0x0008
#define LF_EIGENVALUE_OF_STRUCTURE_TENSOR   0x0010

using namespace vigra;

class ctFeatureExtractor
{
public:
    // constructor
    ctFeatureExtractor(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctFeatureExtractor: ";
        
        type = atoi(ini.GetValue(INI_SECTION_FEATURE_EXTRACTION, "type", "0"));
        if (type == 0) {
            type =  LF_GRAYVALUE | 
                    LF_GRADIENT_MAGNITUDE |
                    LF_DIFFERENCE_OF_GAUSSIAN |
                    LF_EIGENVALUE_OF_HESSIAN |
                    LF_EIGENVALUE_OF_STRUCTURE_TENSOR;
        }
        
        nFeaturesPerScale = (type & LF_GRAYVALUE != 0) + 
                (type & LF_GRADIENT_MAGNITUDE != 0) + 
                (type & LF_DIFFERENCE_OF_GAUSSIAN != 0) + 
                (type & LF_EIGENVALUE_OF_HESSIAN != 0) * 3 + 
                (type & LF_EIGENVALUE_OF_STRUCTURE_TENSOR != 0) * 3;
        
        getMatrix<float >(ini, INI_SECTION_FEATURE_EXTRACTION, "sigmas", sigmas);
        
        scale = atof(ini.GetValue(INI_SECTION_FEATURE_EXTRACTION, "scale", "3"));
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cerr << prefix << "parameters ->" << std::endl;
        std::cerr << "\t\t\t\t sigmas = " << sigmas;
        std::cerr << "\t\t\t\t scale = " << scale << std::endl;
        std::cerr << "\t\t\t\t nFeaturesPerScale = " << nFeaturesPerScale << std::endl;
        std::cerr << "\t\t\t\t featues = " << type << std::endl;
        std::cerr << "\t\t\t\t\t\t type & LF_GRAYVALUE = " << (type & LF_GRAYVALUE != 0) << std::endl;
        std::cerr << "\t\t\t\t\t\t type & LF_GRADIENT_MAGNITUDE = " << (type & LF_GRADIENT_MAGNITUDE != 0) << std::endl;
        std::cerr << "\t\t\t\t\t\t type & LF_DIFFERENCE_OF_GAUSSIAN = " << (type & LF_DIFFERENCE_OF_GAUSSIAN != 0) << std::endl;
        std::cerr << "\t\t\t\t\t\t type & LF_EIGENVALUE_OF_HESSIAN = " << (type & LF_EIGENVALUE_OF_HESSIAN != 0) << std::endl;
        std::cerr << "\t\t\t\t\t\t type & LF_EIGENVALUE_OF_STRUCTURE_TENSOR = " << (type & LF_EIGENVALUE_OF_STRUCTURE_TENSOR != 0) << std::endl;
    }
    
	// operator
    template<int DIM, class FLOAT >
	void run(MultiArrayView<DIM, FLOAT > data, MultiArray<DIM+1, FLOAT > &features)
    {
        typedef typename MultiArray<DIM+1, FLOAT >::difference_type ShapeF;
        // shape of the feature object
        ShapeF shapeF(data.shape(0), data.shape(1), data.shape(2), sigmas.elementCount() * nFeaturesPerScale);
        features.reshape(shapeF, 0);
        
		int fSt, fDim;
		for (int idxS = 0; idxS < sigmas.elementCount(); idxS++) {
			MultiArrayView<DIM+1, FLOAT > mask4;
			MultiArrayView<DIM, FLOAT > mask3;
            if (verbose)
                std::cerr << prefix << "extract features at scale " << sigmas[idxS] << std::endl;

			fSt = idxS * nFeaturesPerScale; 

			// extract smoothed gray value
			fDim = 1;
			//std::cerr << "bindOuter: " << fSt << std::endl;
//            if (verbose)
//                std::cerr << prefix << "start extracting smoothed gray value" << std::endl;
			vigraGaussianSmoothing3<FLOAT >(data, sigmas[idxS], features.bindOuter(fSt));
			fSt += fDim;			

			// extract gaussian gradient magnitude
			fDim = 1;
			//std::cerr << "bindOuter: " << fSt << std::endl;
//            if (verbose)
//                std::cerr << prefix << "start extracting gaussian gradient magnitude" << std::endl;
			vigraGaussianGradientMagnitude3<FLOAT >(data, sigmas[idxS], features.bindOuter(fSt));
			fSt += fDim;

			// extract difference of gaussian
			fDim = 1;
			//std::cerr << "bindOuter: " << fSt << std::endl;
//            if (verbose)
//                std::cerr << prefix << "start extracting difference of gaussian" << std::endl;
			vigraDifferenceOfGaussian3<FLOAT >(data, sigmas[idxS], 2*sigmas[idxS], features.bindOuter(fSt));
			fSt += fDim;

			// extract eigenvalues of hessian matrix
			fDim = 3;
			//std::cerr << "bindOuter: " << fSt << " " << fSt+fDim << std::endl;
//            if (verbose)
//                std::cerr << prefix << "start extracting eigenvalues of hessian matrix" << std::endl;
			vigraEigenValueOfHessianMatrix3<FLOAT >(data, sigmas[idxS], 
                    features.subarray(ShapeF(0, 0, 0, fSt), ShapeF(data.shape(0), data.shape(1), data.shape(2), fSt+fDim)));
			fSt += fDim;

			// extract eigenvalues of structure tensor
			fDim = 3;
			//std::cerr << "bindOuter: " << fSt << " " << fSt+fDim << std::endl;
//            if (verbose)
//                std::cerr << prefix << "start extracting eigenvalues of structure tensor" << std::endl;
			vigraEigenValueOfStructureTensor3<FLOAT >(data, sigmas[idxS], scale, 
                    features.subarray(ShapeF(0, 0, 0, fSt), ShapeF(data.shape(0), data.shape(1), data.shape(2), fSt+fDim)));
			fSt += fDim;
        }
    }

private:
    unsigned short type;
    unsigned short nFeaturesPerScale;
    Matrix<float > sigmas;
    float scale;
    int verbose;
    std::string prefix;
};

#endif /* __CT_FEATURE_EXTRACTOR__ */