#include <iostream>
#include <string>

#include <vigra/timing.hxx>
#include "ArgParser.hxx"
#include "ctIniConfiguration.hxx"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "ctBlockProcessor.hxx"

int main(int argc, char* argv[]) {
    typedef unsigned short INT;
    typedef float FLOAT;
    
    // print help
    if (argc == 1) {
        // print syntax 1
        std::cerr << "Syntax 1: ./segmentation_block_processor " << std::endl;
        std::cerr << "\t\tfile=[ini file]" << std::endl;
        std::cerr << "\t\tstart=[first time step]" << std::endl;
        std::cerr << "\t\tend=[last time step (excluded)]" << std::endl;
        std::cerr << "\t\tmethod=[gc or msa]" << std::endl;
        std::cerr << "\t\tnum_threads=[number of threads to use]" << std::endl;
        std::cerr << "\t\tverbose=[verbose mode]" << std::endl;
        std::cerr << "\t\tcache=[export intermediate results to hdf5 file]" << std::endl;

        std::cerr << "Example for syntax 1: ./segmentation_block_processor file=example.ini start=0 end=5 method=msa num_threads=8 verbose=0 cache=0 " << std::endl << std::endl << std::endl;
        
        // print syntax 2
        std::cerr << "Syntax 2: ./segmentation_block_processor " << std::endl;
        std::cerr << "\t\tfile=[ini file]" << std::endl;
        std::cerr << "\t\ttime=[selected time steps]" << std::endl;
        std::cerr << "\t\tmethod=[gc or msa]" << std::endl;
        std::cerr << "\t\tnum_threads=[number of threads to use]" << std::endl;
        std::cerr << "\t\tverbose=[verbose mode]" << std::endl;
        std::cerr << "\t\tcache=[export intermediate results to hdf5 file]" << std::endl;

        std::cerr << "Example for syntax 2: ./segmentation_block_processor file=example.ini time=\"0,1,2,3,4\" method=msa num_threads=8 verbose=0 cache=0 " << std::endl << std::endl << std::endl;
        
        return 0;
    }
    
    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String, "File that contains the decription for the parallel job.");
    p.addOptionalArg("start", ArgParser::String, "Specify start time.");
    p.addOptionalArg("end", ArgParser::String, "Specify end time (upperbound, excluded from the processing).");
    p.addOptionalArg("time", ArgParser::String, "Specify selected time steps.");
    p.addOptionalArg("method", ArgParser::String, "The method to use for segmentation (gc or msa).");
    p.addOptionalArg("verbose", ArgParser::String, "Use verbose mode.");
    p.addOptionalArg("cache", ArgParser::String, "Cache intermediate results.");
    p.addOptionalArg("num_threads", ArgParser::String, "Number of threads to use.");
    p.parse();
    
    // name of the job file that will be processed
    std::string fileini = (std::string)p.arg("file");
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(fileini.c_str());
    
    // get the time steps to process
    int timeSt, timeEd;
    std::vector<int > times;
    if (p.hasKey("start")) {
        timeSt = atoi(((std::string)p.arg("start")).c_str());
        if (p.hasKey("end")) 
            timeEd = atoi(((std::string)p.arg("end")).c_str());
        else
            timeEd = timeSt + 1;
        
        for (int time = timeSt; time < timeEd; time ++) 
            times.push_back(time);
    }
    else if (p.hasKey("time")) {
        times = string2integers((std::string)p.arg("time"));
    }
    else {
        std::cerr << "No time step specified, program exit(0)." << std::endl;
        return 0;
    }

    // start the block processing
    if (p.hasKey("method")) 
        method = (std::string)p.arg("method");
    
    if (p.hasKey("verbose")) 
        verbose = atoi(((std::string)p.arg("verbose")).c_str());
    
    if (p.hasKey("num_threads")) 
        num_threads = atoi(((std::string)p.arg("num_threads")).c_str());
    
    if (p.hasKey("cache")) 
        cache = atoi(((std::string)p.arg("cache")).c_str());
    
    for (int i = 0; i < times.size(); i++) {
        int time = times[i];
        std::string fileHDF5 = getHDF5File(time, ini);
        std::cerr << "********start block processing for time " << time << "********" << std::endl;
        ctBlockProcessor blockProc(fileHDF5, fileini);
        blockProc.run<3, INT >();
    }
    
    return 0;
}
