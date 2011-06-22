#ifndef __CT_SEGMENTATION_MSA__
#define __CT_SEGMENTATION_MSA__

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

class ctSegmentationMSA
{
public:
    // constructor
    ctSegmentationMSA(
        const Matrix<float > &scales_, 
        int radiusOpening_, 
        int radiusClosing_, 
        int conn_,
        const Matrix<float > &thresholds_)
    {
        radiusOpening = radiusOpening_;
        
        radiusClosing = radiusClosing_;
        
        scales = scales_;
        
        thresholds = thresholds_;
        
        conn = conn_;
        
        prefix = "ctSegmentationMSA: ";
    }
    
    // constructor: load parameters from the ini file
    ctSegmentationMSA(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctSegmentationMSA: ";
        conn = atoi(ini.GetValue(INI_SECTION_MULTISCALE_ANALYSIS, "conn", "6"));
        radiusClosing = atoi(ini.GetValue(INI_SECTION_MULTISCALE_ANALYSIS, "radius_closing", "1"));
        radiusOpening = atoi(ini.GetValue(INI_SECTION_MULTISCALE_ANALYSIS, "radius_opening", "1"));
        getMatrix<float >(ini, INI_SECTION_MULTISCALE_ANALYSIS, "scales", scales);
        getMatrix<float >(ini, INI_SECTION_MULTISCALE_ANALYSIS, "thresholds", thresholds);
        getMatrix<float >(ini, INI_SECTION_MULTISCALE_ANALYSIS, "cutoff", cutoff);
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;
        std::cout << "\t\t\t\t scales = " << scales;
        std::cout << "\t\t\t\t closing radius = " << radiusClosing << std::endl;        
        std::cout << "\t\t\t\t opening radius = " << radiusOpening << std::endl;
        std::cout << "\t\t\t\t thresholds = " << thresholds << std::endl;
        std::cout << "\t\t\t\t cutoff = " << cutoff << std::endl;
    }
    
    // select proper thresholds based on the mean intensity
    template<int DIM, class TIN >
    int selectThresholds(MultiArrayView<DIM, TIN > data)
    {
        double total = 0;
        double nnz = 0;
        for (int x = 0; x < data.shape(0); x++) {
            for (int y = 0; y < data.shape(1); y++) {
                for (int z = 0; z < data.shape(2); z++) {
                    if (data(x, y, z) == 0)
                        continue;
                    
                    total += data(x, y, z);
                    nnz += 1;
                }
            }
        }
        
        if (nnz == 0) 
            return 0;
            
        int sel = 0;
        for (sel = 0; sel < cutoff.elementCount(); sel++) {
            if (total / nnz < cutoff[sel])
                break;
        }
        
        if (verbose) {
            std::cerr << prefix << "mean intensity = " << total / nnz << "; selected threshold index = " << sel << std::endl;
        }
        
        return sel;
    }
    
	// operator
    template<int DIM, class TIN, class TOUT >
	void run(MultiArrayView<DIM, TIN > data, MultiArray<DIM, TOUT > &seg)
    {
        // reshape seg
        seg.reshape(data.shape(), 0);
        
        // select the proper threshold
        int selT = selectThresholds<DIM, TIN >(data);
    
        typename MultiArray<DIM+1, TOUT >::difference_type shapeMatEV;
        for (int i=0; i<DIM; i++) {
            shapeMatEV[i] = data.shape(i);
        }
        shapeMatEV[DIM] = DIM;
        
        MultiArray<DIM, TIN > smoothed(data.shape());
        MultiArray<DIM, bool > seeds(data.shape(), true);
        // walk through scales
        for (int iScale=0; iScale < scales.size(); iScale ++) {
            if (verbose) {
                std::cerr << prefix << "analyze at scale = " << scales[iScale] << std::endl;
            }

            //std::clock_t start = std::clock();
            vigraGaussianSmoothing3<TIN >(data, scales[iScale], smoothed);
            //std::cerr << "smoothing completed at scale " << scales[iScale] << std::endl;
            //std::cerr << ( ( std::clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;

            //start = std::clock();
            MultiArray<DIM+1, TIN > matEigenValues(shapeMatEV);
            vigraEigenValueOfHessianMatrix3<TIN >(smoothed, 0.9, matEigenValues);
            //std::cerr << "eigen values of hessian computed at scale " << scales[iScale] << std::endl;
            //std::cerr << ( ( std::clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
            
            //start = std::clock();
            for (int i=0; i<data.shape(0); i++) {
                for (int j=0; j<data.shape(1); j++) {
                    for (int k=0; k<data.shape(2); k++) {
                        if (!seeds(i, j, k))
                            continue;

                        if (matEigenValues(i, j, k, 0) > thresholds[selT*DIM + 0]) {
                            seeds(i, j, k) = false;
                            continue;
                        }
                         
                        if (matEigenValues(i, j, k, 1) > thresholds[selT*DIM + 1]) {
                            seeds(i, j, k) = false;
                            continue;
                        }
                        
                        if (matEigenValues(i, j, k, 2) > thresholds[selT*DIM + 2]) {
                            seeds(i, j, k) = false;
                            continue;
                        }
                    }
                }
            }
            //std::cerr << ( ( std::clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
        }
        
        // closing
        if (radiusClosing > 0) {
            multiBinaryDilation(srcMultiArrayRange(seeds), destMultiArray(seeds), radiusClosing);
            multiBinaryErosion(srcMultiArrayRange(seeds), destMultiArray(seeds), radiusClosing);
        }
        
        // opening
        if (radiusOpening > 0) {
            multiBinaryErosion(srcMultiArrayRange(seeds), destMultiArray(seeds), radiusOpening);
            multiBinaryDilation(srcMultiArrayRange(seeds), destMultiArray(seeds), radiusOpening);
        }
        
        for (int i=0; i<seeds.elementCount(); i++) {
            if (seeds[i])
                seg[i] = 1;
        }
        
        // cc analysis, if necessary
        if (verbose) {
            std::cerr << prefix << "start connected component analysis " << std::endl;
        }
        if (conn == 6) 
            labelVolumeWithBackground(
                    srcMultiArrayRange(seeds), 
                    destMultiArray(seg),
                    NeighborCode3DSix(), 
                    0);
        else if (conn == 26) 
            // the following code is added due to the bug in vigra::labelVolumeWithBackground function
            for (int x=0; x<seeds.shape(0); x++) {
                for (int y=0; y<seeds.shape(1); y++) {
                    seeds(x, y, 0) = 0;
                    seeds(x, y, seeds.shape(2)-1) = 0;
                }
            }
            
            for (int z=0; z<seeds.shape(2); z++) {
                for (int y=0; y<seeds.shape(1); y++) {
                    seeds(0, y, z) = 0;
                    seeds(seeds.shape(0)-1, y, z) = 0;
                }
            }
            
            for (int z=0; z<seeds.shape(2); z++) {
                for (int x=0; x<seeds.shape(0); x++) {
                    seeds(x, 0, z) = 0;
                    seeds(x, seeds.shape(1)-1, z) = 0;
                }
            }
            // end of the code
            
            labelVolumeWithBackground(
                    srcMultiArrayRange(seeds), 
                    destMultiArray(seg),
                    NeighborCode3DTwentySix(), 
                    0);
    }

private:
    Matrix<float > scales;
    int radiusOpening; 
    int radiusClosing;
    int conn;
    Matrix<float > thresholds;
    Matrix<float > cutoff;
    int verbose;
    std::string prefix;
};

#endif /* __CT_SEGMENTATION_MSA__ */
