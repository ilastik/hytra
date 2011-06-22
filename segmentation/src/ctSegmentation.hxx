#ifndef __CT_SEGMENTATION__
#define __CT_SEGMENTATION__

#include <iostream>
#include <vector>
#include <ctime>
#include "ctSegmentationMSA.hxx"
#include "ctSegmentationSeededWatershed.hxx"
#include "ctAutomatedVoxelClassification.hxx"
#include "ctGVFGenerator.hxx"
#include "ctBoundaryCueGenerator.hxx"
#include "ctSegmentationGC.hxx"
#include "ctSizeFilter.hxx"
#include "ctPreProcessor.hxx"

using namespace vigra;

#define SEGMENTATION_METHOD_MSA             "msa"
#define SEGMENTATION_METHOD_GC              "gc"

template <int DIM, class FLOAT, class INT > void doSegmentation(
        const CSimpleIniA &ini, 
        MultiArray<DIM, FLOAT > &data,
        MultiArray<DIM, INT > &seg, 
        const std::string& method, 
        int verbose = false,
        int cache = false)
{
    // what method to use: 1 - msa only; 2 - graph cut
    //int method = atoi(ini.GetValue(INI_SECTION_RUNTIME, "method", "2"));
    
	// initialize seg
    seg.reshape(data.shape(), 0);
        
    // verbose ??
//    int verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
//    int cache = atoi(ini.GetValue(INI_SECTION_RUNTIME, "cache", "0"));
    std::string prefix = "ctSegmentation: ";
    
    // export the raw data
    if (cache)
        hdf5Write<DIM, FLOAT >(data, "test_segmentation_single_export.h5", "/", "raw");
    
    // pre-processing, e.g. smoothing
    ctPreProcessor preProc(ini, verbose);
    preProc.run<DIM, FLOAT >(data);
    
    // export the raw data
    if (cache)
        hdf5Write<DIM, FLOAT >(data, "test_segmentation_single_export.h5", "/", "smoothed");
    
    // perform multiscale analysis - seeds generated
    ctSegmentationMSA segMSA(ini, verbose);
    MultiArray<DIM, INT > seeds;
    segMSA.run<DIM, FLOAT, INT >(data, seeds);
    if (verbose) {
        std::cerr << prefix << "After multiscale analysis" << std::endl;
        FindMinMax<INT > minmax;
        inspectMultiArray(srcMultiArrayRange(seeds), minmax);
        std::cerr << prefix << "\t\t min = " << minmax.min << std::endl;
        std::cerr << prefix << "\t\t max = " << minmax.max << std::endl;
    }
    
    // perform size filtering
    ctSizeFilter sizeFtr(ini, verbose);
    sizeFtr.run<DIM, INT >(seeds);
    FindMinMax<INT > minmaxSeeds;
    inspectMultiArray(srcMultiArrayRange(seeds), minmaxSeeds);
    if (verbose) {
        std::cerr << prefix << "After size filtering " << std::endl;
        std::cerr << prefix << "\t\t min = " << minmaxSeeds.min << std::endl;
        std::cerr << prefix << "\t\t max = " << minmaxSeeds.max << std::endl;
    }
    if (cache)
        hdf5Write<DIM, INT >(seeds, "test_segmentation_single_export.h5", "/", "seeds");
    
    if (minmaxSeeds.max == 0)
        return ;
	else if (minmaxSeeds.max == 1 || method.compare(SEGMENTATION_METHOD_MSA) == 0) {
        seg.copy(seeds);
        std::replace_if(seg.begin(), seg.end(), std::bind2nd(std::not_equal_to<INT >(), 0), 1);
        return ;
    }
    
    // perform seeded watershed
    ctSegmentationSeededWatershed segSW(ini, verbose);
    MultiArray<DIM, INT > watersheds;
    segSW.run<DIM, FLOAT, INT >(data, seeds, watersheds);
    if (verbose) {
        std::cerr << prefix << "After seeded watershed" << std::endl;
        FindMinMax<INT > minmax;
        inspectMultiArray(srcMultiArrayRange(watersheds), minmax);
        std::cerr << prefix << "\t\t min = " << minmax.min << std::endl;
        std::cerr << prefix << "\t\t max = " << minmax.max << std::endl;

        FindSum<INT > sum;
        inspectMultiArray(srcMultiArrayRange(watersheds), sum);
        std::cerr << prefix << "\t\t total number of watersheds = " << sum() << std::endl;
    }

    if (cache)
        hdf5Write<DIM, INT >(watersheds, "test_segmentation_single_export.h5", "/", "watersheds");

    // automated voxel classification
    ctAutomatedVoxelClassification autoVC(ini, verbose);
    MultiArray<DIM, INT > mask; // mask: 0 - undetermined, 1 - fixed foregound, 2 - fixed background
    MultiArray<DIM, FLOAT > probmap;
    autoVC.run<DIM, INT, FLOAT >(data, seeds, watersheds, mask, probmap);
    
    if (verbose) {
        std::cerr << prefix << "After voxel classification" << std::endl;
        FindMinMax<FLOAT > minmax;
        inspectMultiArray(srcMultiArrayRange(probmap), minmax);
        std::cerr << prefix << "\t\t min = " << minmax.min << std::endl;
        std::cerr << prefix << "\t\t max = " << minmax.max << std::endl;
    }

    if (cache) {
        hdf5Write<DIM, FLOAT >(probmap, "test_segmentation_single_export.h5", "/", "probmap");
        hdf5Write<DIM, INT >(mask, "test_segmentation_single_export.h5", "/", "mask");
    }

    // gvf generation
    ctGVFGenerator gvfGen(ini, verbose);
    MultiArray<DIM+1, FLOAT > gvf;
    gvfGen.run<DIM, INT, FLOAT >(seeds, gvf);
    if (verbose) {
        std::cerr << prefix << "After gvf generation, gvf object size = " << gvf.shape() << std::endl;
    }

    if (cache)
        hdf5Write<DIM+1, FLOAT >(gvf, "test_segmentation_single_export.h5", "/", "gvf");

    // boundary cue generation
    ctBoundaryCueGenerator bdGen(ini, verbose);
    MultiArray<DIM, FLOAT > bdcue;
    bdGen.run<DIM, INT, FLOAT >(watersheds, bdcue);

    if (cache)
        hdf5Write<DIM, FLOAT >(bdcue, "test_segmentation_single_export.h5", "/", "bdcue");

    // gc regularization
    ctSegmentationGC segGC(ini, verbose);
    segGC.run<DIM, FLOAT, INT >(data, mask, probmap, gvf, bdcue, seg);

    if (cache)
        hdf5Write<DIM, INT >(seg, "test_segmentation_single_export.h5", "/", "seg");
}

template <int DIM, class FLOAT, class INT > void doSegmentation(
        const std::string &fileini, 
        MultiArray<DIM, FLOAT > &data,
        MultiArray<DIM, INT > &seg, 
        const std::string& method, 
        int verbose = false,
        int cache = false)
{
    // load ini file
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(fileini.c_str());
    
    // call the segmentation
    doSegmentation<DIM, FLOAT, INT >(ini, data, seg, method, verbose, cache);
}

#endif /* __CT_SEGMENTATION_GC__ */
