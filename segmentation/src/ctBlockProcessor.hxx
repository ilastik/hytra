#ifndef __CT_BLOCK_PROCESSOR__
#define __CT_BLOCK_PROCESSOR__

#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <vigra/multi_array.hxx>
#include "ctIniConfiguration.hxx"
#include "MatReaderWriter.hxx"
#include "HDF5ReaderWriter.hxx"
#include "ctSegmentation.hxx"
#include "ctBlockProcessorThread.hxx"

using namespace vigra;

typedef MultiArray<3, float >::difference_type Shape;
typedef std::pair<Shape, Shape > Task;

/*
 * global variables
 */
 
// mutex and task stack
boost::mutex io_mutex, task_mutex;
std::vector<Task > task_stack;
int task_stack_total;

// memory leak test
bool memory_leak_test = false;

// runtime parameters
int verbose = 0;
int cache = 0;
int num_threads = 1;
std::string method = SEGMENTATION_METHOD_GC;

struct ctBlockProcessorThread {
    // members
    int overlap;
    std::string fileHDF5, dataVariable, dataGroup, segGroup, segVariable;
    std::string fileini;
    int cpuId;
    Shape shape;
    std::string prefix;
    
    // constructor
    ctBlockProcessorThread(int cpuId, const std::string &fileHDF5, const std::string &fileini)
    : cpuId(cpuId), fileHDF5(fileHDF5), fileini(fileini) {
        prefix = "ctBlockProcessorThread: ";
        
        if (!memory_leak_test) {
            boost::mutex::scoped_lock lock(io_mutex);

            CSimpleIniA ini;
            ini.SetUnicode();
            ini.LoadFile(fileini.c_str());
            
            // load data from hdf5 file
            dataVariable = ini.GetValue(INI_SECTION_DATA, "data_variable", "volume");
            dataGroup = ini.GetValue(INI_SECTION_DATA, "data_group", "/raw");
            
            // export information
            segVariable = ini.GetValue(INI_SECTION_DATA, "segmentation_variable", "");
            if (segVariable.size() == 0)
                segVariable = method;
            segGroup = ini.GetValue(INI_SECTION_DATA, "segmentation_group", "/segmentation");
            
            // block parameters
            overlap = atoi(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_overlap", "10"));
            
            // volume size
            shape = hdf5GetDatasetSize<3, unsigned short >(fileHDF5.c_str(), dataGroup.c_str(), dataVariable.c_str());
        }
    }
    
    // print all tasks
    void print() {
    }
    
    // thread operator
    void operator()() {
        while (1) {
            if (!memory_leak_test) {        // process data normally
                Task task; 
                {                    
                    //boost::mutex::scoped_lock lock(task_mutex);
                    boost::mutex::scoped_lock lock1(task_mutex);
                    boost::mutex::scoped_lock lock2(io_mutex);
                    if (task_stack.size() == 0) 
                        return ;
                    
                    task = task_stack.back();
                    task_stack.pop_back();
                    
                    int completed = task_stack_total - task_stack.size();
                    if (completed % 4 == 0)
                        std::cerr << floor (0.5+completed/float(task_stack_total)*100.0) << "%-";
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
                MultiArray<3, float > data; 
                {
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
                doSegmentation<3, float, unsigned short >(fileini, data, seg, method, verbose, cache);

                // save to hdf5 file
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    
                    Shape bottomleftRel = bottomleft - bottomleftEx;
                    Shape toprightRel = bottomleftRel + topright - bottomleft;

                    vigra::HDF5File hdf5file(fileHDF5, true);
                    hdf5file.cd(segGroup);
                    MultiArray<3, unsigned short > tmp(toprightRel - bottomleftRel);
                    tmp.copy(seg.subarray(bottomleftRel, toprightRel));
                    hdf5file.write_block(segVariable, bottomleft, tmp);
                }
            }
//            else {                  // memory leak test: read the same data again
//                // load data
//                MultiArray<3, float > data;
//                data.reshape(Shape(200, 200, 200), 0); 
//                {
//                    boost::mutex::scoped_lock lock(io_mutex);
//                    
//                    std::ifstream ifs("volume_200x200x200_serialized.txt");
//                    unsigned int p;
//                    if (ifs) {
//                        int i = 0;
//                        while (!ifs.eof()) {
//                            ifs >> data[i++];
//                        }
//                    }
//                    
//                    std::cerr << prefix << "cpu " << cpuId << " working ..." << std::endl;
//                }
//                
//                // call the segmentation wrapper
//                MultiArray<3, unsigned short > seg;
//                doSegmentation<3, float, unsigned short >(fileini, data, seg);
//            }
        }
        
        std::cerr << std::endl;
    }
};

class ctBlockProcessor
{
public:
    // constructor: from ini file
    ctBlockProcessor(const std::string &fileHDF5, 
        const std::string &fileini) : 
        fileHDF5(fileHDF5), fileini(fileini)
    {
        prefix = "doBlockProcessing: ";
        
        // ini file
        CSimpleIniA ini;
        ini.SetUnicode();
        ini.LoadFile(fileini.c_str());
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // hdf5 settings
        hdf5Compression = atoi(ini.GetValue(INI_SECTION_RUNTIME, "hdf_compression", "4"));
        hdf5Trunk = atoi(ini.GetValue(INI_SECTION_RUNTIME, "hdf_trunk", "200"));
        
        // load data from hdf5 file
        dataVariable = ini.GetValue(INI_SECTION_DATA, "data_variable", "volume");
        dataGroup = ini.GetValue(INI_SECTION_DATA, "data_group", "/raw");
        
        // export information
        segVariable = ini.GetValue(INI_SECTION_DATA, "segmentation_variable", "");
        if (segVariable.size() == 0)
            segVariable = method;
        segGroup = ini.GetValue(INI_SECTION_DATA, "segmentation_group", "/segmentation");
        
        // block parameters
        blockOverlap = atoi(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_overlap", "5"));
        shapeBlock = string2shape<3 >(ini.GetValue(INI_SECTION_BLOCK_PROCESSING, "block_size", "200, 200, 180"));
        
        // number of threads
//        nThreads = atoi(ini.GetValue(INI_SECTION_RUNTIME, "threads", "1"));
        
        // print parameters
        print();
    }
        
    void print() 
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
        std::cerr << "\t\t\t\t method = " << method  << std::endl;
        std::cerr << "\t\t\t\t num_threads = " << num_threads  << std::endl;
        std::cerr << "\t\t\t\t verbose = " << verbose  << std::endl;
        std::cerr << "\t\t\t\t cache = " << cache  << std::endl;
    }
    
	// operator
    template<int DIM, class INT > void 
        run()
    {
        if (!memory_leak_test) {        // do block processing normally
            boost::mutex::scoped_lock lock1(io_mutex);
            boost::mutex::scoped_lock lock2(task_mutex);
            
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
            
            // create the task stack
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

                        task_stack.push_back(std::make_pair(bottomleft, topright));
                    }
                }
            }
            
            // total number of tasks
            task_stack_total = task_stack.size();
            std::cerr << prefix << "totally " << task_stack_total << " blocks" << std::endl;
            std::cerr << prefix << "completed: ";
        }
        else {                          // in memory leak detection mode
        }

        // intialize threads and start them
        std::vector<boost::thread * > threads;
        boost::thread_group threadGroup;
        for (int iThread = 0; iThread < num_threads; iThread ++) {
            boost::thread *thrd = new boost::thread(ctBlockProcessorThread(iThread, fileHDF5, fileini));
            threads.push_back(thrd);
            threadGroup.add_thread(thrd);
        }
        threadGroup.join_all();  // runs all threads!
        {
            boost::mutex::scoped_lock lock(io_mutex);
            std::cerr << std::endl << prefix << "all thread finished" << std::endl;
            
            // to be on the safe side, delete the threads created by the 'new' operator
            for (int iThread = 0; iThread < num_threads; iThread ++) {
                boost::thread *thrd = threads[iThread];
                threadGroup.remove_thread(thrd);
                delete thrd;
            }
        }
    }

private:
    std::string prefix;

    // ini setting
    std::string fileini;
//    int verbose;
    
    // hdf5 settings
    std::string fileHDF5;    
    int hdf5Compression, hdf5Trunk;
    
    // load data from hdf5 file
    std::string dataVariable, dataGroup;
    
    // export information
    std::string segVariable, segGroup;
    
    // block parameters
    int blockOverlap;
    Shape shapeBlock;
    
    // multi-threading
//    int nThreads;
};



#endif /* __CT_BLOCK_PROCESSOR__ */
