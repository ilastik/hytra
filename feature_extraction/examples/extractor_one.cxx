#include <iostream>
#include <string>

#include <vigra/timing.hxx>

#include <dFeatureExtractor.hxx>
#include <segmentationLabeling.hxx>

// library for parsing input arguments
#include <ArgParser.h>



int main(int argc, char* argv[]){

    //USETICTOC;
    //TIC;


    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String,"File to process");
    p.parse();

    // name of the hdf5 file that will be processed
    std::string filename = (std::string)p.arg("file");

    bool verbose = true;

    // label the segmentation (binary)
    TaskConfiguration tc (filename, 200, verbose);
    FileConfiguration fc ("gc","labels",200,2);
#ifdef WITH_SEGMENTATION_LABELING
    segmentationLabeling(fc,tc);
#endif

    // initialize the dFeatureExtractor
    features::dFeatureExtractor f (filename, features::FT_ALL, verbose);

    // extract features
    f.extract_features(features::FT_ALL);

    //std::cout << TOC << std::endl;

	return 0;
}

