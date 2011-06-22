#include <iostream>
#include <string>

#include <vigra/timing.hxx>
#include "SimpleIni.h"
#include "ArgParser.hxx"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "ctSegmentationRoutine.hxx"

boost::mutex io_mutex;
char buf[1024];

struct TaskWorker
{
    typedef std::pair<int, std::string > Task;
	// members
	int cpuId;
    std::string method;
    std::string inifile;
    std::vector<Task > tasks;

	// constructor
	TaskWorker(int cpuId, std::string inifile, std::vector<Task > tasks) : cpuId(cpuId), tasks(tasks) 
	{
		boost::mutex::scoped_lock lock(io_mutex);
		std::cerr << "cpu " << cpuId << " initialized: # of tasks = " << tasks.size()  << std::endl;
	}

	// constructor
	TaskWorker(int cpuId, std::string inifile) : cpuId(cpuId), inifile(inifile)
	{
		boost::mutex::scoped_lock lock(io_mutex);
		std::cerr << "cpu " << cpuId << " initialized" << std::endl;
	}
    
    // add a new task
    void add(int frame, std::string section) 
    {
        tasks.push_back(std::make_pair(frame, section));
    }
    
    // print all tasks
    void print() 
    {
        std::cerr << "Tasks for CPU " << cpuId << std::endl;
        for (int iTask = 0; iTask < tasks.size(); iTask ++) {
            std::cerr << tasks[iTask].first << ", " << tasks[iTask].second << std::endl;
        }
    }
    
	// thread operator
	void operator()() 
    {
        for (int iTask = 0; iTask < tasks.size(); iTask ++) {
	        {
                boost::mutex::scoped_lock lock(io_mutex);
                std::cerr << "cpu " << cpuId << ": starting working on frame " << tasks[iTask].first << std::endl;
            }
            
            // split data
            split_export<3, unsigned short, unsigned short >(cpuId, tasks[iTask].first, tasks[iTask].second, inifile, io_mutex);
            
            //interpolate_segment_filter
            interpolate_segment_filter<3, unsigned short, unsigned short >(cpuId, tasks[iTask].first, tasks[iTask].second, inifile, io_mutex);
		        //boost::mutex::scoped_lock lock(io_mutex);
                //std::cerr << "\t\tinterpolate_segment_filter completed for frame: " << frame << std::endl;
        }
    }
};


void doSegmentation( const std::string inifile ) {   
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(inifile.c_str());
    const char * pVal;
    
    // intialize task works
    int nThreads = atoi(ini.GetValue("GLOBAL", "threads", "1"));
    std::vector<TaskWorker > workers;
	for (int iWorker = 0; iWorker < nThreads; iWorker ++) {
        workers.push_back(TaskWorker(iWorker, inifile));
        std::cerr << "create a new worker: " << iWorker << std::endl;
    }

    // walk through all sections
    int iCPU = 0;
    for (int iSection = 0; true; iSection ++) {
        sprintf(buf, "TASK%02d", iSection);
        std::string strSection(buf);
        int frameStart = atoi(ini.GetValue(strSection.c_str(), "frame_start", "-1"));
        int frameEnd = atoi(ini.GetValue(strSection.c_str(), "frame_end", "-1"));
        
        std::cerr << "section = " << strSection << ", frameStart = " << frameStart << ", frameEnd = " << frameEnd << std::endl;
        
        if (frameStart == -1 || frameEnd == -1)
            break;
        
        for (int iFrame = frameStart; iFrame < frameEnd; iFrame ++) {
            workers[iCPU].add(iFrame, strSection);
            std::cerr << "    add a new task to cpu " << iCPU << ": " << iFrame << ", " << strSection << std::endl;
            iCPU = (iCPU + 1) % nThreads;
        }
    }
    
    
    std::clock_t start = std::clock();
  	// initialize all the threads
	boost::thread_group threads;
	for (int iWorker = 0; iWorker < nThreads; iWorker ++) {
        //workers[iWorker].print();
		boost::thread *thrd = new boost::thread(workers[iWorker]);
        threads.add_thread(thrd);
	}
	
	threads.join_all();  // runs all threads!

    std::cerr << std::endl << "!!!!!!!!total computation time = " << ( ( std::clock() - start ) / (double)CLOCKS_PER_SEC ) << std::endl;

}


int main(int argc, char* argv[]){
    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String, "File that contains the decription for the parallel job.");	p.parse();

    // name of the job file that will be processed
    std::string inifile = (std::string)p.arg("file");
    doSegmentation( inifile );
    return 0;
}
