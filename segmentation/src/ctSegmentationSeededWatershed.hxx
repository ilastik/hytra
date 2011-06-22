#ifndef __CT_SEGMENTATION_SEEDED_WATERSHED__
#define __CT_SEGMENTATION_SEEDED_WATERSHED__

#include <iostream>
#include <vector>
#include <ctime>
#include <vigra/seededregiongrowing3d.hxx>

using namespace vigra;

class ctSegmentationSeededWatershed
{
public:
    // constructor: load parameters from the ini file
    ctSegmentationSeededWatershed(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        prefix = "ctSegmentationSeededWatershed: ";
        if (atoi(ini.GetValue(INI_SECTION_SEEDED_WATERSHED, "type", "1")) == 1)
            type = vigra::KeepContours;
        else
            type = vigra::CompleteGrow;
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
        
    // print parameters
    void print()
    {
        std::cerr << prefix << " parameters -> " << std::endl;
        if (type == vigra::KeepContours)
            std::cout << "\t\t\t\t type = vigra::KeepContours" << std::endl;
        else
            std::cout << "\t\t\t\t type = vigra::CompleteGrow" << std::endl;
    }

	// operator
    template<int DIM, class TIN, class TOUT > void run(
            MultiArrayView<DIM, TIN > data, 
            MultiArrayView<DIM, TOUT > seeds, 
            MultiArray<DIM, TOUT > &seg)
    {
        // initialize output
        seg.reshape(seeds.shape(), 0);
        
        // get max label number
        FindMinMax<TOUT > minmaxSeeds;
        inspectMultiArray(srcMultiArrayRange(seeds), minmaxSeeds);
        TOUT max_region_label = static_cast<TOUT >(minmaxSeeds.max);
        
        // get max intensity
        FindMinMax<TIN > minmaxData;
        inspectMultiArray(srcMultiArrayRange(data), minmaxData);
        MultiArray<DIM, TIN > dataInv;
        dataInv.reshape(data.shape(), minmaxData.max);
        dataInv /= data;

        // call seeded region growing
        if (verbose) {
            std::cerr << prefix << "start seeded region growing" << std::endl;
        }
        ArrayOfRegionStatistics<vigra::SeedRgDirectValueFunctor<TOUT > > gradstat(max_region_label);
        seededRegionGrowing3D(srcMultiArrayRange(dataInv), srcMultiArray(seeds), destMultiArray(seg), 
                gradstat, vigra::KeepContours, NeighborCode3DSix(), -1.0);
        
        // only preserve the watersheds, if using vigra::KeepContours
        if (type == vigra::KeepContours) {
            std::replace_if(seg.begin(), seg.end(), std::bind2nd(std::not_equal_to<TOUT >(), 0), 2);
            std::replace_if(seg.begin(), seg.end(), std::bind2nd(std::equal_to<TOUT >(), 0), 1);
            std::replace_if(seg.begin(), seg.end(), std::bind2nd(std::equal_to<TOUT >(), 2), 0);
        }
    }

private:
    SRGType type;
    std::string prefix;
    int verbose;
};

#endif /* __CT_SEGMENTATION_SEEDED_WATERSHED__ */
