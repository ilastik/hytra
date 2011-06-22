#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

#include <vigra/timing.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_array.hxx>

// library for parsing input arguments
#include <ArgParser.h>

#include <ConfigMpi.hxx>
#include <ConfigFeatures.hxx>


std::vector<int> readIndexList(std::string filename)
{
    std::ifstream ifile (filename.c_str());
    std::vector<int> res;

    while(!ifile.eof())
    {
        int input;
        ifile >> input;
        res.push_back(input);
    }

    return res;
}

void recipeStatistics(std::string filename)
{


    Recipe r = readRecipe(filename);


    for(int t = 0; t < r.numTasks; t++){

        std::string statfilename = constructFilename("statistics.txt",r.tasks[0].dataFolder);
        std::ofstream ofile (statfilename.c_str());

        std::vector<int> cellNumber (r.tasks[0].numJobs);
        std::vector<int> cellProduction (r.tasks[0].numJobs,0);
        std::vector<double> cellVolume (r.tasks[0].numJobs);
        std::vector<double> cellIntensity (r.tasks[0].numJobs);

        for(int i = 0; i < r.tasks[0].numJobs; i++){

            std::string fn = constructFilename(r.tasks[0].files[i], r.tasks[0].dataFolder);

            std::cout << fn << "\t";
            vigra::HDF5File f (fn, vigra::HDF5File::Open);
            f.cd("/features");
            feature_array numCells (array_shape(1));
            f.read("supervoxels",numCells);
            const int numLabels = numCells[0];

            array_shape shape (numLabels);
            feature_array labelContent (shape);

            f.read("labelcontent",labelContent);


            int cellCount = 0;
            double volCount = 0;
            double intCount = 0;
            double intVar = 0;
            double posX = 0;
            double posY = 0;
            double posZ = 0;


            for(int j = 1; j <= numLabels; j++){
                if(labelContent[j-1] != 0){
                    cellCount ++;
                    std::stringstream sstr;
                    sstr <<"/features/" << j;
                    f.cd(sstr.str());

                    feature_array volume (array_shape(1));
                    f.read("volume",volume);

                    feature_array intensity (array_shape(4));
                    f.read("intensity",intensity);

                    feature_array position (array_shape(12));
                    f.read("position",position);

                    posX += position[0];
                    posY += position[1];
                    posZ += position[2];

                    volCount += volume[0];
                    intCount += intensity[0];
                    intVar += intensity[1];

                }
            }
            if(cellCount > 0){
                volCount /= cellCount;
                intCount /= cellCount;
                intVar /= cellCount;
                posX /= cellCount;
                posY /= cellCount;
                posZ /= cellCount;
            }

            cellNumber[i] = cellCount;
            cellVolume[i] = volCount;
            cellIntensity[i] = intCount;

            if(i>0){
                cellProduction[i] = cellNumber[i] - cellNumber[i-1];
            }

            ofile << i << "\t" << cellNumber[i] << "\t" << cellVolume[i] << "\t"  << cellIntensity[i] << "\t" << intVar << "\t"  << posX << "\t"  << posY << "\t"  << posZ << "\n" ;

            ofile.flush();
            std::cout << " - OK\n";
        }
        std::cout << "File written to " << statfilename << "\n";
    }

}


void fileStatistics(std::string filename, std::string indices = "all")
{
    vigra::HDF5File h5 (filename, vigra::HDF5File::Open);
    int maxlabel;
    h5.readAtomic("/features/supervoxels",maxlabel);

    std::cout << "MaxLabel: " << maxlabel << "\n";

    std::vector<int> ind;
    if(indices != "all")
    {
        ind = readIndexList(indices);
    }
    else
    {
        vigra::ArrayVector<hsize_t> shape = h5.getDatasetShape("/features/labelcontent");
        label_array labelContent = label_array(array_shape(shape[0]));
        h5.read("/features/labelcontent",labelContent);

        for(int i = 0; i < maxlabel; i++)
        {
            if(labelContent[i] != 0)
            {
                ind.push_back(i+1);
            }
        }
    }

    // file with lists of undersegmented, oversegmented and correct cells
    /*std::ofstream file_us (std::string(filename + ".us.txt").c_str());
    std::ofstream file_os (std::string(filename + ".os.txt").c_str());
    std::ofstream file_ok (std::string(filename + ".ok.txt").c_str());
    /**/
    std::ofstream file (std::string(filename + ".txt").c_str());

    for(int i = 0; i < ind.size(); i++)
    {
        std::vector<feature_type> features;
        std::stringstream group;
        group << "/features/" << ind[i];

        h5.cd(group.str());
        features.push_back(ind[i]);

        feature_array volume (array_shape(1));
        h5.read("volume",volume);
        features.push_back(volume[0]);

        feature_array intensity (array_shape(4));
        h5.read("intensity",intensity);
        for(int j = 0; j < 2; j++)
            features.push_back(intensity[j]);

        feature_array pc (array_shape(12));
        h5.read("pc",pc);
        for(int j = 0; j < 3; j++)
            features.push_back(pc[j]);

        feature_array pos (array_shape(12));
        h5.read("position",pos);
        feature_array com (array_shape(12));
        h5.read("com",com);
        double displacement = std::sqrt(std::pow(com[0]-pos[0],2)+std::pow(com[1]-pos[1],2)+std::pow(com[2]-pos[2],2));
        if(!isnan(displacement))
        {
            features.push_back(displacement);
        }
        else
        {
            features.push_back(0);
        }


        /*
          ...
          */


        // select a file according to the cell being under/oversegmented or
        // correctly segmented.
        std::ofstream *ofile;
        /*if(volume[0] < 20){
            ofile = &file_os;
        } else if(volume[0] > 450){
            ofile = &file_us;
        }else{
            ofile = &file_ok;
        }/**/
        ofile = &file;

        // write out all features of a cell in one line
        for(int j = 0; j < features.size(); j++)
        {
            *ofile << features[j];
            if(j != features.size()-1)
                *ofile << "\t";
            else
                *ofile << "\n";
        }


    }



    std::cout << "File(s) written to " << filename << ".*\n";


}

#include "interfaces.hxx"
int main(int argc, char* argv[]){
    // parse arguments
    ArgParser p(argc, argv);
    p.addRequiredArg("file", ArgParser::String,
        "File that contains the decription for the parallel job.");
    p.addRequiredArg("type", ArgParser::String,
                     "Type of analysis performed: 'recipe', 'file', 'track'.");
    p.addOptionalArg("indices", ArgParser::String,
                     "File that contains a list of cell indices.");
    p.parse();

    // parse arguments
    std::string filename = (std::string)p.arg("file");


    std::string type = (std::string)p.arg("type");
    std::string indices = "all";
    if(p.hasKey("indices")){
        indices = (std::string)p.arg("indices");
    }


    if(type == "recipe"){
        recipeStatistics(filename);
    }

    if(type == "file"){
        fileStatistics(filename, indices);
    }

    if(type == "track"){

    }


	return 0;
}

