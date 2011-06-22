
#ifndef MPIMASTERSLAVE_HXX
#define MPIMASTERSLAVE_HXX

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <boost/mpi.hpp>

#include <ConfigMpi.hxx>

#include <dFeatureExtractor.hxx>
#include <segmentationLabeling.hxx>


// start a Master, NodeMaster or Worker according to the rank and the ppn
void runParallel(boost::mpi::communicator comm, // mpi world communicator
                 int ppn,   // number of processes per node
                 std::string filename, // job file
                 bool FOR_HELICS  // run code for helics
                 );


class MpiMaster {
public:
    // Constructor. Provide a vector containing the ranks of the node masters
    // and the path to the recipe file.
    MpiMaster(boost::mpi::communicator comm, IntList nodeMasterList, std::string recipeFile, bool FOR_HELICS);

	int run();


private:
	typedef std::vector<std::string> Strings;
    typedef std::vector<NodeMasterInfo> StatusList;
	typedef std::vector<JobInfo> JobStatusList;

    // Helics configuration
    bool FOR_HELICS_;
    
    // MPI communicator
    boost::mpi::communicator comm_;

    // full description of the tasks, including configurations, files, etc.
    Recipe recipe_;

    // number of NodeMasters
    int numNodeMasters_;
    IntList nodeMasterRank_;

    // stores the job index that was assigned to a NodeMaster
    IntList jobAssignment_;

    // list of NodeMasters' status
    StatusList nodeMasterStatus_;
    int numAliveNodeMasters_();

    // check for answers from NodeMasters
    /*bool checkMail_( int &fromNm, int *message );*/

    // give a job to a certain NodeMaster
    void sendJob_( int nmRank);

    // mark the job of NodeMaster as done, postprocess
    void finalizeJob_( int nmRank , JobStatus exitStatus );

    // return the NodeMaster index given its rank.
    int findNodeMasterFromRank_(int rank);

	// create a detailed report
	void createReport_(int taskIndex);

    // save/load the current processing status of the files for resuming
    void setStatus_(int task, int file, JobStatus jstatus);
    std::string getStatus_(int task, int file);
};


/** MpiNodeMaster runs and manages jobs on one node of Helics.
  * May have several MpiWorkers as support for calculations
  */
class MpiNodeMaster {
public:
    MpiNodeMaster(boost::mpi::communicator comm, int rank, IntList workers, std::string recipeFile, bool FOR_HELICS);
    int run();

private:
    // Helics configuration
    bool FOR_HELICS_;

    // MPI communicator
    boost::mpi::communicator comm_;
    
    // number of workers, worker ranks
    int numWorkers_;
    IntList workers_;

    // full description of the tasks, including configurations, files, etc.
    Recipe recipe_;

    // current Task index
    int currentTaskIndex_;

    // rank of NodeMaster (for identification in verbose mode)
    int rank_;

    // current NodeMaster state
    Status status_;

    // do the job
    void doJob_( std::string filename );

    // copy a dataset to local temp directory. Returns full path to file
    std::string getFile_( std::string filename );

    // copy the dataset from temp directory to home directory
    void sendFile_( std::string filename );

};



/** MpiWorker has no real functionality by now
  *
  */
class MpiWorker {
public:
    MpiWorker(boost::mpi::communicator comm, int nodeMaster);
    void run();

private:
    // MPI communicator
    boost::mpi::communicator comm_;

    // rank of parent NodeMaster
    int nodeMaster_;

};





#endif //MPIMASTERSLAVE_HXX
