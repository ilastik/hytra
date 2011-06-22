
#include "MpiMasterSlave.hxx"


void runParallel(boost::mpi::communicator world, int ppn, std::string filename, bool FOR_HELICS = false)
{
    int rank = world.rank();
    int size = world.size();
    // rank 0: run as Master
    if(rank == 0)
    {
        // get the ranks of all NodeMasters
        IntList nodeMasters (1,1);
        for(int i = ppn; i < size; i+= ppn)
        {
            if(i != 1)
            {
                nodeMasters.push_back(i);
            }
        }
        MpiMaster master (world, nodeMasters, filename, FOR_HELICS);
        master.run();
    }
    // rank 1: run as NodeMaster
    else if(rank == 1)
    {
        IntList workers;
        for(int i = rank+1; i < rank+ppn-1 && i < size; i+= 1)
        {
            workers.push_back(i);
        }
        MpiNodeMaster nmaster (world, rank, workers, filename, FOR_HELICS);
        nmaster.run();
    }
    // run as NodeMaster if process is the first on this node
    else if(rank % ppn == 0)
    {
        IntList workers;
        for(int i = rank+1; i < rank+ppn && i < size; i+= 1)
        {
            workers.push_back(i);
        }
        MpiNodeMaster nmaster (world, rank, workers, filename, FOR_HELICS);
        nmaster.run();
    }
    // otherwise run as Worker
    else
    {
        int nodemaster = (rank / ppn)*ppn;
        if (nodemaster == 0)
        {
            nodemaster = 1;
        }
        MpiWorker worker (world, nodemaster);
        worker.run();
    }
}




MpiMaster::MpiMaster(boost::mpi::communicator comm, std::vector<int> nodeMasterList, std::string reciepeFile, bool FOR_HELICS = false)
{
    //MPI communicator
    comm_ = comm;

    // evaluate recipe file
    recipe_ = readRecipe(reciepeFile);
    recipe_.startTime = MPI::Wtime();

    // store NodeMasters' ranks
    numNodeMasters_ = nodeMasterList.size();
    nodeMasterRank_ = nodeMasterList;

    // initialize information about NodeMasters
    NodeMasterInfo init;
    init.status = Status_Idle;
    init.taskIndex = 0;
    nodeMasterStatus_ = StatusList (numNodeMasters_, init);

    // set job assignments to default
    jobAssignment_ = IntList (numNodeMasters_, -1);

    // Execute code for helics
    FOR_HELICS_ = FOR_HELICS;
}

int MpiMaster::run()
{
    bool exit = false;
    // Stay in this loop until all jobs are done
    while( !exit ){
        // check mail
        boost::optional<boost::mpi::status> optional =
                comm_.iprobe(boost::mpi::any_source, boost::mpi::any_tag);

        if(optional){
            boost::mpi::status mpiStatus = *optional;
            int source = mpiStatus.source();
            Msg message;
            // recieve message
            comm_.recv(source,boost::mpi::any_tag,message);

            // handle the different messages
            switch(message)
            {
            case Msg_AskForJob:
                sendJob_(source);
                break;

            case Msg_JobFinished:
                // mark job as done
                finalizeJob_(source, JobStatus_Done);
                // send new job if possible
                sendJob_(source);
                break;

            case Msg_Error:
                // mark job as erroneous
                finalizeJob_(source, JobStatus_Error);
                // send new job if possible
                sendJob_(source);
                break;

            default:
                std::cerr << "MpiMaster: Recieved unknown message from NodeMaster " << source << ".\n";
                break;
            }
        }

        // count the number of open tasks
        int sumOpenJobs = 0;
        for(int t = 0; t < recipe_.numTasks; t++)
        {
            sumOpenJobs += recipe_.tasks[t].numOpenJobs;
        }

        if(sumOpenJobs == 0 && numAliveNodeMasters_() == 0)
        {
            exit = true;
        }
    }


    for(int t = 0; t < recipe_.numTasks; t++)
    {
        createReport_(t);
	}

	return 0;

}


void MpiMaster::sendJob_( int nmRank )
{

    int nm = findNodeMasterFromRank_(nmRank);

    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiMaster: Sending instructions to NodeMaster " << nodeMasterRank_[nm] << ".\n";
	}

    int found = -1;
    int taskIndex = 0;
    bool exit = false;
    while(!exit)
    {
        int i = 0;
        // search for a job in current task
        for(JobStatusList::iterator it ( recipe_.tasks[taskIndex].jobStatus.begin() ); it != recipe_.tasks[taskIndex].jobStatus.end(); ++it, ++i)
        {
            if(it->status == JobStatus_Undone)
            {
                if(getStatus_(taskIndex,i) == jobStatusToString(JobStatus_Done))
                {
                    recipe_.tasks[taskIndex].numOpenJobs--;
                    it->status = JobStatus_Done;
                    it->duration = -1;
                }
                else
                {
                    found = i;
                    break;
                }
            }
        }

        if(found != -1)
        {
            exit = true; // exit if something was found
        }
        else
        {
            if(taskIndex < recipe_.numTasks-1)
            {
                taskIndex++; //otherwise look in the next task
            }
            else
            {
                exit = true; // or exit if there are no more tasks
            }
        }
    }


    if( found == -1 && taskIndex == recipe_.numTasks-1 )
    {
        // nothing found and last task --> time to send NodeMaster sleeping
        comm_.send(nmRank,Tag_Message,Msg_Shutdown);
        nodeMasterStatus_[nm].status = Status_Offline;
    }
    else
    {
        // send the task index to NodeMaster
        nodeMasterStatus_[nm].taskIndex = taskIndex;
        comm_.send(nmRank,Tag_TaskIndex,taskIndex);

        // send the filename to NodeMaster
        comm_.send(nmRank,Tag_DataTransfer,recipe_.tasks[taskIndex].files[found]);

        // mark NodeMaster and job as buisy
        jobAssignment_[nm] = found;
        nodeMasterStatus_[nm].status = Status_Working;

        recipe_.tasks[taskIndex].jobStatus[found].status = JobStatus_InProgress;
        recipe_.tasks[taskIndex].jobStatus[found].startTime =
                MPI::Wtime() - recipe_.startTime;
        recipe_.tasks[taskIndex].jobStatus[found].nodeMaster = nm;
        setStatus_(taskIndex,found,JobStatus_InProgress);

	}

	return;
}

void MpiMaster::finalizeJob_( int nmRank, JobStatus exitStatus = JobStatus_Done )
{
    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiMaster: NodeMaster " << nmRank << " finished his job.\n";
	}

    int nm = findNodeMasterFromRank_(nmRank);

    int t = nodeMasterStatus_[nm].taskIndex;
    // set the job as done and the NodeMaster as free
    recipe_.tasks[t].numOpenJobs--;
    recipe_.tasks[t].jobStatus[jobAssignment_[nm]].status = exitStatus;
    recipe_.tasks[t].jobStatus[jobAssignment_[nm]].duration =
            MPI::Wtime() - recipe_.tasks[t].jobStatus[jobAssignment_[nm]].startTime - recipe_.startTime;

    setStatus_(t,jobAssignment_[nm],exitStatus);
    jobAssignment_[nm] = -1;
    nodeMasterStatus_[nm].status = Status_Idle;

	return;
}

int MpiMaster::numAliveNodeMasters_()
{
	int count = 0;
    for(int i = 0; i < nodeMasterStatus_.size(); i++)
    {
        // count nodemasters that are still awake
        if(nodeMasterStatus_[i].status != Status_Offline)
        {
			count++;
		}
	}
    return count;
}

int MpiMaster::findNodeMasterFromRank_(int rank)
{

    int nodeMaster = -1;
    for(int i = 0; i < nodeMasterRank_.size(); i++)
    {
        if(nodeMasterRank_[i] == rank)
        {
            nodeMaster = i;
            break;
        }
    }
    return nodeMaster;
}


void MpiMaster::setStatus_(int task, int file, JobStatus jstatus)
{
    std::stringstream fname;
    fname << recipe_.name << "_status.txt";
    std::string filename = constructFilename(fname.str(), recipe_.tasks[task].dataFolder);

    CSimpleIniA statusFile;
    statusFile.SetUnicode();
    statusFile.LoadFile(filename.c_str());

    std::stringstream taskname;
    taskname << "Task" << task;
    std::string status = jobStatusToString(jstatus);
    statusFile.SetValue(taskname.str().c_str(), recipe_.tasks[task].files[file].c_str(), status.c_str());
    statusFile.SaveFile(filename.c_str());

    return;
}

std::string MpiMaster::getStatus_(int task, int file)
{
    std::stringstream status;

    std::stringstream fname;
    fname << recipe_.name << "_status.txt";
    std::string filename = constructFilename(fname.str(), recipe_.tasks[task].dataFolder);

    CSimpleIniA statusFile;
    statusFile.SetUnicode();
    statusFile.LoadFile(filename.c_str());

    std::stringstream taskname;
    taskname << "Task" << task;
    status << statusFile.GetValue(taskname.str().c_str(), recipe_.tasks[task].files[file].c_str(), "");
    statusFile.SaveFile(filename.c_str());

    return status.str();
}

void MpiMaster::createReport_(int taskIndex)
{

	// use the data folder
	std::stringstream fname;
    fname << recipe_.name << "_task" << taskIndex << ".txt";
    std::string filename =
           constructFilename(fname.str(), recipe_.tasks[taskIndex].dataFolder);

	std::ofstream outfile (filename.c_str(), std::ios::trunc);

	// get number of successful jobs
	int successful = 0;
    for(int i=0; i<recipe_.tasks[taskIndex].jobStatus.size(); i++)
    {
        if(recipe_.tasks[taskIndex].jobStatus[i].status == JobStatus_Done)
        {
			successful ++;
		}
	}

	outfile << "------------------------------------------------------\n";
    outfile << "   Report - " << recipe_.name << " - Task "<<taskIndex << "\n";
    outfile << "------------------------------------------------------\n\n";

    outfile << "Description: " << recipe_.description << "\n\n\n";

	outfile << "Working paths:\n";
	outfile << "--------------\n\n";
    outfile << "- Data path: " << recipe_.tasks[taskIndex].dataFolder << "\n";
    outfile << "- Temp path: " << recipe_.tempFolder << "\n\n\n";

	outfile << "Job summary:\n";
	outfile << "----------------\n\n";
    outfile << "- Number of NodeMasters:     " << numNodeMasters_ << "\n";
    outfile << "- Number of jobs:       " <<
            recipe_.tasks[taskIndex].files.size() << "\n";
	outfile << "  - Successful:         " << successful << "\n";
    outfile << "  - Failed:             " <<
            recipe_.tasks[taskIndex].files.size() - successful << "\n";
    outfile << "- Total execution time: " <<
            MPI::Wtime() - recipe_.startTime << " s\n\n\n";

	outfile << "Jobs with errors:\n";
	outfile << "-------------------------\n\n";
    for(int i = 0; i < recipe_.tasks[taskIndex].jobStatus.size(); i++)
    {
        if(recipe_.tasks[taskIndex].jobStatus[i].status != JobStatus_Done)
        {
			outfile << "- Job " << i+1 << ":\n";
            outfile << "  - Filename:    " <<
                    recipe_.tasks[taskIndex].files[i] << "\n";
            outfile << "  - Start time:  " <<
                    recipe_.tasks[taskIndex].jobStatus[i].startTime << " s\n";
            outfile << "  - Duration:    " <<
                    recipe_.tasks[taskIndex].jobStatus[i].duration << " s\n";
            outfile << "  - NodeMaster index: " <<
                    recipe_.tasks[taskIndex].jobStatus[i].nodeMaster << "\n";
            outfile << "  - Exit status: " <<
                    jobStatusToString(recipe_.tasks[taskIndex].jobStatus[i].status) << "\n\n";
		}
	}
	outfile << "\n";

	outfile << "Jobs without errors:\n";
	outfile << "-------------------------\n\n";
    for(int i = 0; i < recipe_.tasks[taskIndex].jobStatus.size(); i++)
    {
        if(recipe_.tasks[taskIndex].jobStatus[i].status == JobStatus_Done)
        {
			outfile << "- Job " << i+1 << ":\n";
            outfile << "  - Filename:    " <<
                    recipe_.tasks[taskIndex].files[i] << "\n";
            outfile << "  - Start time:  " <<
                    recipe_.tasks[taskIndex].jobStatus[i].startTime << " s\n";
            outfile << "  - Duration:    " <<
                    recipe_.tasks[taskIndex].jobStatus[i].duration << " s\n";
            outfile << "  - NodeMaster index: " <<
                    recipe_.tasks[taskIndex].jobStatus[i].nodeMaster << "\n";
            outfile << "  - Exit status: " <<
                    jobStatusToString(recipe_.tasks[taskIndex].jobStatus[i].status) << "\n\n";
		}
	}

	outfile.close();

	return ;
}




MpiNodeMaster::MpiNodeMaster(boost::mpi::communicator comm, int rank, IntList workers, std::string recipeFile, bool FOR_HELICS = false)
{
    comm_ = comm;
    recipe_ = readRecipe(recipeFile);
    recipe_.startTime = MPI::Wtime();
	rank_ = rank;
	currentTaskIndex_ = 0;
    workers_ = workers;
    numWorkers_ = workers_.size();

    // execute code for helics
    FOR_HELICS_ = FOR_HELICS;
}

int MpiNodeMaster::run()
{
    // ask for a job
    comm_.send(0,Tag_Message,Msg_AskForJob);


    // Stay in this loop until all jobs are done
	bool exit = false;
    while( !exit )
    {
        // check mail
        boost::optional<boost::mpi::status> optional =
                comm_.iprobe(boost::mpi::any_source, boost::mpi::any_tag);

        if(optional){
            boost::mpi::status mpiStatus = *optional;
            int source = mpiStatus.source();
            Tag tag = (Tag)mpiStatus.tag();

            // handle the different messages
            switch(tag)
            {
            case Tag_Message:
                {
                    // recieve a message
                    Msg message;
                    comm_.recv(source,tag,message);
                    if(message == Msg_Shutdown)
                    {
                        exit = true;
                    }
                    break;
                }

            case Tag_TaskIndex:
                {
                    // recieve a task index
                    comm_.recv(source,tag,currentTaskIndex_);
                    break;
                }

            case Tag_DataTransfer:
                {
                    // recieve a filename
                    std::string filename;
                    comm_.recv(source,Tag_DataTransfer,filename);

                    std::string jname;
                    if(FOR_HELICS_){
                        jname = getFile_(filename);
                    }else{
                        jname = filename;
                    }

                    if(jname == "")
                    {
                        comm_.send(0,Tag_Message,Msg_Error);
                    }
                    else
                    {
                        doJob_(jname);
                    }
                    break;
                }

            default:
                std::cerr << "MpiNodeMaster "<< rank_ << ": Error - recieved unknown message from master.\n";
                break;

            }
        }
    }


    // send shutdown signal to workers
    for(int i = 0; i < numWorkers_; i++)
    {
        comm_.send(workers_[i],Tag_Message,Msg_Shutdown);
    }

    std::cout << "MpiNodeMaster "<< rank_  << " finished.\n";

}


std::string MpiNodeMaster::getFile_( std::string filename )
{
    std::string fromFile = constructFilename(filename, recipe_.tasks[currentTaskIndex_].dataFolder);
    std::string toFile = constructFilename(filename, recipe_.tempFolder);

	// force copying from home to local temp
	std::string systemCommand ("cp -f " + fromFile + " " + toFile);
	if (std::system(systemCommand.c_str()) != 0){
        std::cerr << "MpiNodeMaster "<< rank_
                  << ": Failed to copy file: " << filename << "\n";
		return "";
    }

    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiNodeMaster "<< rank_ << ": Copied file " << filename << " to temp directory.\n";
	}
	return toFile;
}

void MpiNodeMaster::sendFile_( std::string filename )
{
    std::string fromFile = constructFilename(filename, recipe_.tempFolder);
    std::string toFile = constructFilename(filename, recipe_.tasks[currentTaskIndex_].dataFolder);

    // force moving from local temp to home
    std::string systemCommand ("mv -f " + fromFile + " " + toFile);
    if( std::system(systemCommand.c_str()) != 0)
    {
        throw(0);
        return;
    }

    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiNodeMaster "<< rank_ << ": Copied file " << filename << " back to home directory.\n";
    }
    return;
}


void MpiNodeMaster::doJob_( std::string filename )
{
    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiNodeMaster "<< rank_
                  << ": Start processing file " << filename << ".\n";
	}

    // set the correct filename of the local file to process
    std::string localFilename;
    if(FOR_HELICS_){
        localFilename = filename;
    }else{
        localFilename = constructFilename(filename, recipe_.tasks[currentTaskIndex_].dataFolder );
    }


    try
    {
        bool verbose = recipe_.verbosity == Vb_Loud;

        //----------------------------------------------------------------------
        // All caclulations have to be placed here.
        //----------------------------------------------------------------------

        // label the segmentation (binary)
        TaskConfiguration tc (localFilename, recipe_.tasks[currentTaskIndex_].blockSize, verbose);
        FileConfiguration fc ("gc","labels",200,2);
#ifdef WITH_SEGMENTATION_LABELING
        segmentationLabeling(fc,tc);
#endif

        // initialize the dFeatureExtractor
        features::dFeatureExtractor f (localFilename, features::FT_ALL, verbose);

        // extract features
        f.extract_features(features::FT_ALL);

        if(FOR_HELICS_){
            // send back the processed file
            sendFile_(localFilename);
        }
	}
    catch(...)
    {
        std::cerr << "MpiNodeMaster " << rank_ << ": error processing file " << filename << "\n";
        comm_.send(0,Tag_Message,Msg_Error);
		return;
	}

    if(recipe_.verbosity == Vb_Loud || recipe_.verbosity == Vb_MpiMessages)
    {
        std::cout << "MpiNodeMaster "<< rank_ << ": Job finished.\n";
	}

	// send the done signal
    comm_.send(0,Tag_Message,Msg_JobFinished);

	return;
}





MpiWorker::MpiWorker(boost::mpi::communicator comm, int nodeMaster)
{
    nodeMaster_ = nodeMaster;
    comm_ = comm;
}

void MpiWorker::run()
{
    // Stay in this loop until all jobs are done
    bool exit = false;
    while( !exit )
    {
        // check mail
        boost::optional<boost::mpi::status> optional =
                comm_.iprobe(boost::mpi::any_source, boost::mpi::any_tag);

        if(optional)
        {
            boost::mpi::status mpiStatus = *optional;
            int source = mpiStatus.source();
            Tag tag = (Tag)mpiStatus.tag();

            // handle the different messages
            switch(tag)
            {
            case Tag_Message:
                {
                    // recieve a message
                    Msg message;
                    comm_.recv(source,tag,message);
                    if(message == Msg_Shutdown)
                    {
                        exit = true;
                    }
                    break;
                }

            default:
                std::cerr << "MpiWorker: Error - recieved unknown message from master.\n";
                break;


            }
        }
    }
}


