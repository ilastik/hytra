
#ifndef CONFIGMPI_HXX
#define CONFIGMPI_HXX

#include <iostream>
#include <string>
#include <vector>

#include <SimpleIni.h>

enum Msg {
    Msg_AskForJob,
    Msg_Wait,
    Msg_JobFinished,
    Msg_Shutdown,
    Msg_Data,
    Msg_TaskIndex,
    Msg_Error
};

enum Tag {
    Tag_Message,
    Tag_TaskIndex,
    Tag_DataTransfer
};

enum Status {
    Status_Idle,
    Status_Working,
    Status_Offline
};

enum JobStatus {
    JobStatus_Done,
    JobStatus_InProgress,
    JobStatus_Undone,
    JobStatus_Error
};

enum Verbosity {
	Vb_None,
    Vb_Silent,
    Vb_MpiMessages,
    Vb_Loud
};



typedef std::vector<int> IntList;


// Information about a NodeMaster
struct NodeMasterInfo {
    Status status;
    int taskIndex;
};

// Information about a single job
struct JobInfo {
	JobStatus status;
	double startTime;
	double duration;
    int nodeMaster;
};

// Information about a task
struct TaskInfo {
	std::string dataFolder;

	int blockSize;

	typedef std::vector<std::string> Strings;
	Strings files;

	typedef std::vector<JobInfo> JobStatusList;
	JobStatusList jobStatus;

	// number of (open) jobs
	int numJobs;
	int numOpenJobs;
};


// Whole recipe for processing several Tasks
struct Recipe {
	// Short name and description for the report
	std::string name;
	std::string description;

    // local temp folder for the NodeMasters
	std::string tempFolder;

	// sets the amount of information printed
	Verbosity verbosity;

	// start time of the master
	double startTime;

	// number of tasks
	int numTasks;

	// information about each task
	typedef std::vector<TaskInfo> TaskList;
	TaskList tasks;

};

static Recipe readRecipe(std::string filename){
    Recipe recipe;
    CSimpleIniA recipeFile;
    recipeFile.SetUnicode();
    recipeFile.LoadFile(filename.c_str());

    // read the recipe info part
    recipe.name = std::string( recipeFile.GetValue("Recipe", "Name", "DefaultJobName") );
    recipe.description = std::string( recipeFile.GetValue("Recipe", "Description", "") );
    recipe.tempFolder = std::string( recipeFile.GetValue("Recipe", "SlaveTempPath", "/tmp/tempdata/") );


    std::string read_temp ( recipeFile.GetValue("Recipe", "Verbosity", "MpiOnly") );
	if(read_temp == "MpiOnly"){
        recipe.verbosity = Vb_MpiMessages;
	}else if(read_temp == "Silent"){
        recipe.verbosity = Vb_Silent;
	}else if(read_temp == "None"){
        recipe.verbosity = Vb_None;
	}else{
        recipe.verbosity = Vb_Loud;
	}

	std::stringstream readss;
    readss << recipeFile.GetValue("Recipe", "NumberOfTasks", "1");
    readss >> recipe.numTasks;
    recipe.tasks.resize(recipe.numTasks);

	// read each task
    for(int t = 0; t < recipe.numTasks; t++){
		std::stringstream currentTaskName;
		currentTaskName << "Task" <<  t;

		// read data path for this task
        recipe.tasks[t].dataFolder = std::string( recipeFile.GetValue(currentTaskName.str().c_str(), "DataPath", "/home/user/tempdata/") );

		// read the files
        std::string taskType ( recipeFile.GetValue(currentTaskName.str().c_str(), "Type", "") );
		if(taskType == "List"){
			// get number of files
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "NumberOfFiles", "0");
            readss >> recipe.tasks[t].numJobs;
            recipe.tasks[t].numOpenJobs = recipe.tasks[t].numJobs;

			// get the block size for block processing
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "BlockSize", "100");
            readss >> recipe.tasks[t].blockSize;

			// read the filenames one by one
            for(int i = 0; i < recipe.tasks[t].numJobs; i++){
				std::stringstream ident;
				ident << "File" << i;
                recipe.tasks[t].files.push_back( std::string(recipeFile.GetValue(currentTaskName.str().c_str(), ident.str().c_str(), "")) );
				JobInfo tempInfo;
				tempInfo.status = JobStatus_Undone;
				tempInfo.startTime = -1;
				tempInfo.duration = -1;
                tempInfo.nodeMaster = -1;
                recipe.tasks[t].jobStatus.push_back(tempInfo);
			}
		}else if(taskType == "Template"){
			// get the block size for block processing
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "BlockSize", "100");
            readss >> recipe.tasks[t].blockSize;

			// get the time frame range
			int from, to, step;
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "FromTimeFrame", "0");
			readss >> from;
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "ToTimeFrame", "0");
			readss >> to;
			readss.clear();
            readss << recipeFile.GetValue(currentTaskName.str().c_str(), "TimeFrameStep", "1");
			readss >> step;

            std::string nameTemplate ( recipeFile.GetValue(currentTaskName.str().c_str(), "Template", "") );

			// construct the filenames one by one
			int count = 0;
			for(int i = from; i <= to; i += step){
				count++;
				char tempname [255];
				sprintf(tempname,nameTemplate.c_str(),i);
                recipe.tasks[t].files.push_back( std::string( tempname ) );
				JobInfo tempInfo;
				tempInfo.status = JobStatus_Undone;
				tempInfo.startTime = -1;
				tempInfo.duration = -1;
                tempInfo.nodeMaster = -1;
                recipe.tasks[t].jobStatus.push_back(tempInfo);
			}
            recipe.tasks[t].numJobs = count;
            recipe.tasks[t].numOpenJobs = count;
		}else{
            std::cerr << "Error reading recipe file. No task type provided, use 'Task = List' or 'Task = Template'.";
		}
	}


    return recipe;
}

static inline std::string jobStatusToString(JobStatus jStatus){
	switch(jStatus){
	case JobStatus_Done: return "Done"; break;
	case JobStatus_InProgress: return "In progress"; break;
	case JobStatus_Undone: return "Not processed"; break;
	case JobStatus_Error: return "Error"; break;
	default: return "unknown status"; break;
	}
	return "unknown status";
}


static inline std::string messageToString(Msg message){
	switch(message){
	case Msg_AskForJob: return "Asking for job"; break;
	case Msg_Wait: return "Wait for instructions"; break;
	case Msg_JobFinished: return "Job finished"; break;
    case Msg_Shutdown: return "Shutdown signal"; break;
	case Msg_Data: return "Data transfer"; break;
	case Msg_TaskIndex: return "New task index"; break;
	case Msg_Error: return "Error"; break;
	default: return "unknown message"; break;
	}
	return "unknown status";
}


static inline std::string constructFilename(std::string filename, std::string dataFolder) {
	// extract the pure filename (without folders)
	if(filename.rfind("/") != std::string::npos){
		filename = std::string(filename.begin()+filename.rfind("/")+1,filename.end());
	}

	return std::string (dataFolder + filename);
}

#endif //CONFIGMPI_HXX
