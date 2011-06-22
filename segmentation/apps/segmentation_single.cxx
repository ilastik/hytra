#include <iostream>
#include <string>

#include <vigra/timing.hxx>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <vigra/multi_array.hxx>
#include "ArgParser.hxx"
#include "ctIniConfiguration.hxx"
#include "MatReaderWriter.hxx"
#include "HDF5ReaderWriter.hxx"
#include "ctSegmentation.hxx"

typedef MultiArray<3, unsigned int >::difference_type Shape;

Shape bottomleft = string2shape<3 >("0, 0, 0");
Shape topright = string2shape<3 >("0, 0, 0");;
int verbose = 1;
int cache = 1;
int num_threads = 1;
std::string method = SEGMENTATION_METHOD_GC;

template<int DIM, class INT, class FLOAT >
void run(const std::string &fileini, const int time)
{
    std::string prefix = "Main: ";
    
    // ini parser
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(fileini.c_str());
    
//    // verbose ??
//    int verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
//    int hdf5export = atoi(ini.GetValue(INI_SECTION_RUNTIME, "hdf5_export", "0"));
    
    // load data from hdf5 file
    std::string dataVariable = ini.GetValue(INI_SECTION_DATA, "data_variable", "volume");
    std::string dataGroup = ini.GetValue(INI_SECTION_DATA, "data_group", "/raw");
    std::string fileHDF5 = getHDF5File(time, ini);
    if (verbose) {
        std::cerr << prefix << "Read from file " << fileHDF5 << std::endl;
    }
    
	MultiArray<DIM, FLOAT > data;
    Shape shape;
    vigra::HDF5File hdf5file(fileHDF5.c_str(), true);
    if (topright[0] <= 0 || topright[1] <= 0 || topright[2] <= 0) {
	    MultiArray<DIM, INT > raw;
        shape = hdf5GetDatasetSize<DIM, INT >(fileHDF5.c_str(), dataGroup, dataVariable);
        raw.reshape(shape);
        hdf5file.cd(dataGroup);
        hdf5file.read_block(dataVariable, Shape(0, 0, 0), raw.shape(), raw);
        data.reshape(raw.shape());
        data.copy(raw);
    }
    else {
	    MultiArray<DIM, INT > raw;
        shape = topright - bottomleft;
        raw.reshape(shape);
        hdf5file.cd(dataGroup);
        hdf5file.read_block(dataVariable, bottomleft, raw.shape(), raw);
        data.reshape(raw.shape());
        data.copy(raw);
    }
    
    if (verbose) {
        std::cerr << prefix << "Read in data: size = " << data.shape() << std::endl;
        FindMinMax<INT > minmax;
        inspectMultiArray(srcMultiArrayRange(data), minmax);
        std::cerr << prefix << "\t\t min = " << minmax.min << std::endl;
        std::cerr << prefix << "\t\t max = " << minmax.max << std::endl;
    }
    
    // call the segmentation wrapper
    clock_t start = clock();
    MultiArray<3, INT > seg;
    doSegmentation<3, FLOAT, INT >(fileini, data, seg, method, verbose, cache);
    std::cerr << prefix << "total time for segmentation = " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
}

int main(int argc, char* argv[]) {
    typedef unsigned short INT;
    typedef float FLOAT;
    
    std::string prefix = "Main: ";
    
    // print help
    if (argc == 1) {
        // print syntax 
        std::cerr << "Syntax: ./segmentation_single " << std::endl;
        std::cerr << "\t\tfile=[ini file]" << std::endl;
        std::cerr << "\t\ttime=[selected time steps]" << std::endl;
        std::cerr << "\t\tbottomleft=[bottomleft of the selected volume]" << std::endl;
        std::cerr << "\t\ttopright=[topright of the selected volume (excluded)]" << std::endl;
        std::cerr << "\t\tmethod=[gc or msa]" << std::endl;
        std::cerr << "\t\tverbose=[verbose mode]" << std::endl;
        std::cerr << "\t\tcache=[export intermediate results to hdf5 file]" << std::endl;

        std::cerr << "Example: ./segmentation_single file=example.ini time=0 method=msa verbose=1 cache=1 bottomleft=\"385, 585, 0\" topright=\"615, 815, 215\"" << std::endl << std::endl << std::endl;
        
        return 0;
    }
    
    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String, "File that contains the decription for the parallel job.");
    p.addOptionalArg("time", ArgParser::String, "Specify the time.");
    p.addOptionalArg("bottomleft", ArgParser::String, "Specify the bottom-left coordinates.");
    p.addOptionalArg("topright", ArgParser::String, "Specify the top-right coordinates.");
    p.addOptionalArg("method", ArgParser::String, "The method to use for segmentation (gc or msa).");
    p.addOptionalArg("verbose", ArgParser::String, "Use verbose mode.");
    p.addOptionalArg("cache", ArgParser::String, "Cache intermediate results.");
    p.parse();

    // name of the job file that will be processed
    std::string fileini = (std::string)p.arg("file");
    int time;
    if (p.hasKey("time")) 
        time = atoi(((std::string)p.arg("time")).c_str());
    else
        time = 0;

    if (p.hasKey("bottomleft")) 
        bottomleft = string2shape<3 >((std::string)p.arg("bottomleft"));
    
    if (p.hasKey("topright")) 
        topright = string2shape<3 >((std::string)p.arg("topright"));

    if (p.hasKey("method")) 
        method = (std::string)p.arg("method");
    
    if (p.hasKey("verbose")) 
        verbose = atoi(((std::string)p.arg("verbose")).c_str());
    
    if (p.hasKey("cache")) 
        cache = atoi(((std::string)p.arg("cache")).c_str());
    
    std::cerr << prefix << "file = " << fileini << std::endl;
    std::cerr << prefix << "time = " << time << std::endl;
    std::cerr << prefix << "bottomleft = " << bottomleft << std::endl;
    std::cerr << prefix << "topright = " << topright << std::endl;
    std::cerr << prefix << "method = " << method << std::endl;
    std::cerr << prefix << "verbose = " << verbose << std::endl;
    std::cerr << prefix << "cache = " << cache << std::endl;
    std::cerr << prefix << "time = " << time << std::endl;
    
    // call the segmentation
    run<3, INT, FLOAT >(fileini, time);
    
    /*
    // load the entire dataset from hdf5 file
    clock_t start;
	MultiArray<3, unsigned int > raw;
    Shape shape = hdf5GetDatasetSize<3, unsigned int >("/export/home/xlou/export/ct-keller-animal/stack_0099.h5", "/raw", "volume");
    raw.reshape(shape);
	hdf5Read<3, unsigned int >(raw, "/export/home/xlou/export/ct-keller-animal/stack_0099.h5", "/raw", "volume");
    std::cerr << "time for feature extraction = " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
    std::cerr << "size " << raw.shape() << std::endl;
    */
  /*  
    clock_t start;
    vigra::HDF5File hdf5file("/export/home/xlou/export/ct-keller-animal/stack_0099.h5", true);
    Shape offset(10, 200, 200);
    Shape shape(100, 200, 200);
	MultiArray<3, unsigned int > raw;
    raw.reshape(shape);
    hdf5file.cd("/raw");
	hdf5file.read_block("volume", offset, shape, raw);
    std::cerr << "time for data loading = " << ( ( clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;
    std::cerr << "size " << raw.shape() << std::endl;
*/    
    return 0;
}
