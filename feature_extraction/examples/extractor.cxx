#include <iostream>
#include <string>

#include <vigra/timing.hxx>

// MPI support, use boost.MPI
#include <boost/mpi.hpp>

// library for job management
#include <MpiMasterSlave.hxx>

// library for parsing input arguments
#include <ArgParser.h>



namespace mpi = boost::mpi;


int main(int argc, char* argv[]){
    //USETICTOC;
    //TIC;


    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String,
        "File that contains the decription for the parallel job.");
    p.parse();

    // name of the job file that will be processed
    std::string filename = (std::string)p.arg("file");

    // prepare boost.MPI
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if(world.size() < 2) {
        throw std::runtime_error("Error: extractor has been compiled with MPI support, but only one process has been started.\n"
                                 "At least two instances are needed.\n");
        world.abort(1 /*EXIT_FAILURE*/);
    }


    // run master or slave
    runParallel(world, 1, filename, false);


    //std::cout << TOC << std::endl;

	return 0;
}

