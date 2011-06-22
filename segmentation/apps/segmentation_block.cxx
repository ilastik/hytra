#include <iostream>
#include <string>

#include <vigra/timing.hxx>
#include "ArgParser.hxx"
#include "ctIniConfiguration.hxx"
#include "MatReaderWriter.hxx"
#include "HDF5ReaderWriter.hxx"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "ctSegmentation.hxx"
//#include "ctBlockProcessor.hxx"

boost::mutex io_mutex;
boost::mutex task_mutex;
typedef MultiArray<3, float >::difference_type Shape;
typedef std::pair<Shape, Shape > Task;
std::vector<Task > task_stack;
int task_stack_total;

struct ctBlockProcessorThread {
    // members
    int overlap;
    std::string fileHDF5, dataVariable, dataGroup, segGroup, segVariable;
    std::vector<Task > tasks;
    std::string fileini;
    int cpuId;
    Shape shape;
    int verbose;
    std::string prefix;
    
    // constructor
    ctBlockProcessorThread(int cpuId, const std::string &fileHDF5, const std::string &fileini)
    : cpuId(cpuId), fileHDF5(fileHDF5), fileini(fileini) {
        prefix = "ctBlockProcessorThread: ";
        
        CSimpleIniA ini;
        ini.SetUnicode();
        ini.LoadFile(fileini.c_str());
        
        // load data from hdf5 file
        dataVariable = ini.GetValue(INI_SECTION_DATA, "data_variable", "volume");
        dataGroup = ini.GetValue(INI_SECTION_DATA, "data_group", "/raw");
        
        // export information
        segVariable = ini.GetValue(INI_SECTION_DATA, "segmentation_variable", "volume");
        segGroup = ini.GetValue(INI_SECTION_DATA, "segmentation_group", "/segmentation");
        
        // block parameters
        overlap = atoi(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_overlap", "10"));
        
        // volume size
        shape = hdf5GetDatasetSize<3, unsigned short >(fileHDF5.c_str(), dataGroup.c_str(), dataVariable.c_str());
    }
    
    // add a new task
    void add(const Shape &bottomleft, const Shape &topright) {
        tasks.push_back(std::make_pair(bottomleft, topright));
    }
    
    // print all tasks
    void print() {
    }
    
    // thread operator
    void operator()() {
        while (1) {
            Task task; 
            {
                boost::mutex::scoped_lock lock(task_mutex);
                if (task_stack.size() == 0) {
//                    boost::mutex::scoped_lock lock(io_mutex);
//                    std::cerr << prefix << "cpu " << cpuId << " exit ..." << std::endl;
                    return ;
                }
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    int completed = task_stack_total - task_stack.size();
                    if (completed % 4 == 0)
                        std::cerr << floor (0.5+completed/float(task_stack_total)*100.0) << "%-";
                }
                task = task_stack.back();
                task_stack.pop_back();
                {
//                    boost::mutex::scoped_lock lock(io_mutex);
//                    std::cerr << prefix << "cpu " << cpuId << " task_stack size down to " << task_stack.size() << std::endl;
                }
            }

            // add overlaps
            Shape bottomleft = task.first;
            Shape topright = task.second;
            Shape bottomleftEx(
                    std::max<int >(0, bottomleft[0] - overlap),
                    std::max<int >(0, bottomleft[1] - overlap),
                    std::max<int >(0, bottomleft[2] - overlap));
            Shape toprightEx(
                    std::min<int >(topright[0] + overlap, shape[0]),
                    std::min<int >(topright[1] + overlap, shape[1]),
                    std::min<int >(topright[2] + overlap, shape[2]));

            // load data
            MultiArray<3, float > data; {
                boost::mutex::scoped_lock lock(io_mutex);

                MultiArray<3, unsigned short > raw;
                vigra::HDF5File hdf5file(fileHDF5.c_str(), true);
                data.reshape(toprightEx - bottomleftEx);
                raw.reshape(toprightEx - bottomleftEx);
                hdf5file.cd(dataGroup);
                hdf5file.read_block(dataVariable, bottomleftEx, raw.shape(), raw);

                data.copy(raw);
            }

            // call the segmentation wrapper
            MultiArray<3, unsigned short > seg;
            doSegmentation<3, float, unsigned short >(fileini, data, seg);

            // save to hdf5 file
            {
                Shape bottomleftRel = bottomleft - bottomleftEx;
                Shape toprightRel = bottomleftRel + topright - bottomleft;

                boost::mutex::scoped_lock lock(io_mutex);
                vigra::HDF5File hdf5file(fileHDF5, true);
                hdf5file.cd(segGroup);
                MultiArray<3, unsigned short > tmp(toprightRel - bottomleftRel);
                tmp.copy(seg.subarray(bottomleftRel, toprightRel));
                hdf5file.write_block(segVariable, bottomleft, tmp);
            }
        }
        
        std::cerr << std::endl;
    }
};

template<int DIM, class INT >
        void doBlockProcessing(const std::string &fileHDF5, const std::string &fileini) {
    typedef typename MultiArray<DIM, INT >::difference_type Shape;
    
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(fileini.c_str());
    
    std::string prefix = "doBlockProcessing: ";
    
    // verbose ?
    int verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
    
    // hdf5 settings
    int hdf5Compression = atoi(ini.GetValue(INI_SECTION_RUNTIME, "hdf_compression", "4"));
    int hdf5Trunk = atoi(ini.GetValue(INI_SECTION_RUNTIME, "hdf_trunk", "200"));
    
    // load data from hdf5 file
    std::string dataVariable = ini.GetValue(INI_SECTION_DATA, "data_variable", "volume");
    std::string dataGroup = ini.GetValue(INI_SECTION_DATA, "data_group", "/raw");
    
    // export information
    std::string segVariable = ini.GetValue(INI_SECTION_DATA, "segmentation_variable", "volume");
    std::string segGroup = ini.GetValue(INI_SECTION_DATA, "segmentation_group", "/segmentation");
    
    // block parameters
    int blockOverlap = atoi(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_overlap", "5"));
    Shape shapeBlock = string2shape<3 >(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_size", "200, 200, 180"));
    
    // number of threads
    int nThreads = atoi(ini.GetValue(INI_SECTION_RUNTIME, "threads", "1"));
    
    // print parameters
    //if (verbose)
    {
        std::cerr << prefix << "parameters ->" << std::endl;
        std::cerr << "\t\t\t\t fileHDF5 = " << fileHDF5  << std::endl;
        std::cerr << "\t\t\t\t fileini = " << fileini  << std::endl;
        std::cerr << "\t\t\t\t hdf5Compression = " << hdf5Compression  << std::endl;
        std::cerr << "\t\t\t\t hdf5Trunk = " << hdf5Trunk  << std::endl;
        std::cerr << "\t\t\t\t dataVariable = " << dataVariable  << std::endl;
        std::cerr << "\t\t\t\t dataGroup = " << dataGroup  << std::endl;
        std::cerr << "\t\t\t\t segVariable = " << segVariable  << std::endl;
        std::cerr << "\t\t\t\t segGroup = " << segGroup  << std::endl;
        std::cerr << "\t\t\t\t blockOverlap = " << blockOverlap  << std::endl;
        std::cerr << "\t\t\t\t shapeBlock = " << shapeBlock  << std::endl;
        std::cerr << "\t\t\t\t nThreads = " << nThreads  << std::endl;
    }
    
    // block-processing parameters
    Shape shape = hdf5GetDatasetSize<3, INT >(fileHDF5.c_str(), dataGroup.c_str(), dataVariable.c_str());
    Shape blocks(ceil(shape[0]/float(shapeBlock[0])),
            ceil(shape[1]/float(shapeBlock[1])),
            ceil(shape[2]/float(shapeBlock[2])));
    
    std::cerr << prefix << "raw data size = " << shape << std::endl;
    
    // create hdf5 file and enables features
    vigra::HDF5File hdf5file(fileHDF5, true);
    hdf5file.enableChunks(hdf5Trunk);
    hdf5file.enableCompression(hdf5Compression);
    
    // create datasets
    hdf5file.cd_mk(segGroup);
    hdf5file.createDataset(segVariable.c_str(), shape, static_cast<INT >(0));
    std::cerr << prefix << "create dataset " << segVariable << std::endl;
    
    // intialize all threads
    std::vector<ctBlockProcessorThread > threads;
    for (int iThread = 0; iThread < nThreads; iThread ++)
        threads.push_back(ctBlockProcessorThread(iThread, fileHDF5, fileini));
    
    int iThread = 0;
    for (int i=0; i<blocks[0]; i++) {
        for (int j=0; j<blocks[1]; j++) {
            for (int k=0; k<blocks[2]; k++) {
                int x = i*shapeBlock[0];
                int y = j*shapeBlock[1];
                int z = k*shapeBlock[2];
                
                Shape bottomleft(
                        std::max<int >(0, x),
                        std::max<int >(0, y),
                        std::max<int >(0, z));
                Shape topright(
                        std::min<int >(x+shapeBlock[0], shape[0]),
                        std::min<int >(y+shapeBlock[1], shape[1]),
                        std::min<int >(z+shapeBlock[2], shape[2]));
                
                threads[iThread].add(bottomleft, topright);
                //std::cerr << prefix << "add [" << bottomleft << "," << topright << "] to thread " << iThread << std::endl;
                iThread = (iThread + 1) % nThreads;
                
                task_stack.push_back(std::make_pair(bottomleft, topright));
            }
        }
    }

    // total number of tasks
    task_stack_total = task_stack.size();
    {
        boost::mutex::scoped_lock lock(io_mutex);
        std::cerr << prefix << "completed: ";
    }
    
    // initialize all the threads
    boost::thread_group threadGroup;
    for (int iThread = 0; iThread < nThreads; iThread ++) {
        boost::thread *thrd = new boost::thread(threads[iThread]);
        threadGroup.add_thread(thrd);
    }

    threadGroup.join_all();  // runs all threads!
    {
        boost::mutex::scoped_lock lock(io_mutex);
        std::cerr << std::endl << prefix << "all thread finished" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    
    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String, "File that contains the decription for the parallel job.");
    p.addOptionalArg("time", ArgParser::String, "Specify time to process, instead of loading from the ini file.");
    p.parse();
    
    // name of the job file that will be processed
    std::string fileini = (std::string)p.arg("file");
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(fileini.c_str());
    
    int timeSt, timeEd;
    if (p.hasKey("time")) {
        timeSt = atoi(((std::string)p.arg("time")).c_str());
        timeEd = timeSt + 1;
    }
    else {
        timeSt = atoi(ini.GetValue(INI_SECTION_DATA, "frame_start", "0"));
        timeEd = atoi(ini.GetValue(INI_SECTION_DATA, "frame_end", "1"));
    }
    
    for (int time = timeSt; time < timeEd; time ++) {
        std::cerr << "********start block processing for time " << time << "********" << std::endl;
        std::string fileHDF5 = getHDF5File(time, ini);
        doBlockProcessing<3, unsigned short >(fileHDF5, fileini);
    }
    
    return 0;
}
