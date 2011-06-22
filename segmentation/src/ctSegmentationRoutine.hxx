#ifndef __CT_SEGMENTATION_ROUTINE__
#define __CT_SEGMENTATION_ROUTINE__

#include <stdio.h>
#include <iostream>
#include <string>
#include "SimpleIni.h"
#include "ctSegmentationMSA.hxx"
#include "HDF5ReaderWriter.hxx"
#include "MatReaderWriter.hxx"
#include "vigraAlgorithmPackage.hxx"
#include <boost/thread/mutex.hpp>

template<int DIM, class TIN, class TOUT >
void split_export(int cpuId, int frame, std::string section, std::string inifile, boost::mutex &io_mutex) 
{
    typedef typename MultiArray<DIM, TIN >::difference_type Shape;
    /*
     * load ini file
     */
    char buf[1024];
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(inifile.c_str());
    
    /*
     * block-processing parameters
     */
    int blockOverlap = atoi(ini.GetValue(section.c_str(), "block_overlap", "5"));
    Shape sizeRaw = string2shape<DIM >(ini.GetValue(section.c_str(), "size_raw", "300, 300, 200"));
    Shape blockSizeRaw = string2shape<DIM >(ini.GetValue(section.c_str(), "block_size", "300, 300, 200"));
    Shape nBlocks = string2shape<DIM >(ini.GetValue(section.c_str(), "block_count", "4, 4, 3"));
    //std::cerr << "block size: " << blockSizeRaw << std::endl;
    //std::cerr << "block count: " << nBlocks << std::endl;
    
    /*
     * split data and save to temporatory directory
     */
    
    // load the raw data
    std::string strVariable = ini.GetValue(section.c_str(), "infile_variable", "vol");
    std::string strGroup = ini.GetValue(section.c_str(), "infile_group", "/");
    std::string strInFilePath = ini.GetValue(section.c_str(), "infile_path", "/");
    std::string strInFileTemplate = ini.GetValue(section.c_str(), "infile_template", "/");
    sprintf(buf, strInFileTemplate.c_str(), frame);
    std::string strInFile = strInFilePath + std::string("/") + std::string(buf);
    
    MultiArray<DIM, TIN > data;
    {
		boost::mutex::scoped_lock lock(io_mutex);
        if (strInFile.find(".mat") != std::string::npos) {
//            mxReadFromMatFile<DIM, TIN >(data, strInFile.c_str(), strVariable);
        }
        else {
            Shape shape = hdf5GetDatasetSize<DIM, TIN >(strInFile.c_str(), strGroup, strVariable);
            data.reshape(shape);
		    hdf5Read<DIM, TIN >(data, strInFile.c_str(), strGroup, strVariable);
        }
        //std::cerr << "cpu " << cpuId << ": loading from .mat file completed" << std::endl;
    }
    
    // split
//    for (int i=1; i<2; i++) {
//        for (int j=1; j<2; j++) {
//            for (int k=1; k<2; k++) {
    for (int i=0; i<nBlocks[0]; i++) {
        for (int j=0; j<nBlocks[1]; j++) {
            for (int k=0; k<nBlocks[2]; k++) {
                int x = i*blockSizeRaw[0];
                int y = j*blockSizeRaw[1];
                int z = k*blockSizeRaw[2];
                Shape coordBottomLeft(
                    std::max<int >(0, x-blockOverlap), 
					std::max<int >(0, y-blockOverlap), 
					std::max<int >(0, z-blockOverlap));
				Shape coordTopRight(
					std::min<int >(x+blockSizeRaw[0]+blockOverlap, sizeRaw[0]), 
					std::min<int >(y+blockSizeRaw[1]+blockOverlap, sizeRaw[1]), 
					std::min<int >(z+blockSizeRaw[2]+blockOverlap, sizeRaw[2]));

                MultiArrayView<DIM, TIN > dataSub = data.subarray(coordBottomLeft, coordTopRight);
                MultiArray<DIM, TIN > tmp(dataSub.shape());
                tmp.copy(dataSub);
                sprintf(buf, "/tmp/%04d_%d_%d_%d.h5", frame, i, j, k);
		        {
                    boost::mutex::scoped_lock lock(io_mutex);
    			    hdf5Write<DIM, TIN >(tmp, buf, "/", strVariable);
                    //std::cerr << "cpu " << cpuId << ": tmp file written -> " << buf << std::endl;
                }
            }
        }
    }
}

template<int DIM, class TIN, class TOUT >
void interpolate_segment_filter(int cpuId, int frame, std::string section, std::string inifile, boost::mutex &io_mutex) 
{
    typedef typename MultiArray<DIM, TIN >::difference_type Shape;
    /*
     * load ini file
     */
    char buf[1024];
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(inifile.c_str());
    
    /*
     * block-processing parameters
     */
    Shape interp = string2shape<DIM >(ini.GetValue(section.c_str(), "interpolation", "1, 1, 3"));
    Shape sizeRaw = string2shape<DIM >(ini.GetValue(section.c_str(), "size_raw", "300, 300, 200"));
    Shape sizeInterp = string2shape<DIM >(ini.GetValue(section.c_str(), "size_interp", "300, 300, 200"));
    int blockOverlap = atoi(ini.GetValue(section.c_str(), "block_overlap", "5"));
    Shape blockSizeRaw = string2shape<DIM >(ini.GetValue(section.c_str(), "block_size", "300, 300, 200"));
    Shape blockSizeInterp = string2shape<DIM >(ini.GetValue(section.c_str(), "block_size_interp", "300, 300, 200"));
    std::vector<int > nBlocks = string2integers(ini.GetValue(section.c_str(), "block_count", "4, 4, 3"));

    float intensityThreshold = atof(ini.GetValue(section.c_str(), "intensity_threshold", "0"));
    
    std::string strVariable = ini.GetValue(section.c_str(), "infile_variable", "vol");
    int hdfTrunk = atoi(ini.GetValue("GLOBAL", "hdf_trunk", "0"));
    int hdfCompression = atoi(ini.GetValue("GLOBAL", "hdf_compression", "0"));
    
    /*
     * segmentation parameters
     */
    std::vector<float > scales  = string2floats(ini.GetValue(section.c_str(), "msa_scales", "1.2, 2.4"));
    int rClosing = atoi(ini.GetValue(section.c_str(), "msa_closing_radius", "3"));;
    int rOpening = atoi(ini.GetValue(section.c_str(), "msa_opening_radius", "1"));; 
    std::vector<float > thresholds  = string2floats(ini.GetValue(section.c_str(), "msa_thresholds", "-1, -1.5, -2.0"));
    ctSegmentationMSA segMSA(scales, rClosing, rOpening, thresholds);
    {
        boost::mutex::scoped_lock lock(io_mutex);
        //std::cerr << "cpu " << cpuId << ": segmentation algorithm initialized -> " << std::endl;
    }
    
    /*
     * export file
     */
    std::string varRaw = ini.GetValue(section.c_str(), "outfile_variable_raw", "vol");
    std::string groupRaw = ini.GetValue(section.c_str(), "outfile_group_raw", "vol");
    std::string varSegmentation = ini.GetValue(section.c_str(), "outfile_variable_segmentation", "vol");
    std::string groupSegmentation = ini.GetValue(section.c_str(), "outfile_group_segmentation", "vol");
    
    std::string outFilePath = ini.GetValue(section.c_str(), "outfile_path", "/");
    std::string outFileTemplate = ini.GetValue(section.c_str(), "outfile_template", "/");
    sprintf(buf, outFileTemplate.c_str(), frame);
    std::string outFile = outFilePath + std::string("/") + std::string(buf);
    
    // create hdf5 file and enables features
    {
        boost::mutex::scoped_lock lock(io_mutex);
        vigra::HDF5File hdf5file(outFile, false);
        hdf5file.enableChunks(hdfTrunk);
        hdf5file.enableCompression(hdfCompression);
        
        // create datasets
        hdf5file.cd_mk(groupRaw);
        hdf5file.createDataset(varRaw, sizeInterp, static_cast<TIN >(0));
        hdf5file.cd_mk(groupSegmentation);
        hdf5file.createDataset(varSegmentation, sizeInterp, static_cast<TOUT >(0));
        
        //std::cerr << "cpu " << cpuId << ": hdf5 file initialized -> " << outFile << std::endl;
    }
    //std::cerr << "hdf5 and datasets created: " << outFile << std::endl;
    
    /*
     * load tmp data and process
     */
//    for (int i=1; i<2; i++) {
//        for (int j=1; j<2; j++) {
//            for (int k=1; k<2; k++) {
    for (int i=0; i<nBlocks[0]; i++) {
        for (int j=0; j<nBlocks[1]; j++) {
            for (int k=0; k<nBlocks[2]; k++) {
                int x = i*blockSizeRaw[0];
                int y = j*blockSizeRaw[1];
                int z = k*blockSizeRaw[2];
                
                Shape coordBottomLeftInterp(
                    interp[0] * std::max<int >(0, x), 
                    interp[1] * std::max<int >(0, y), 
                    interp[2] * std::max<int >(0, z));
                Shape coordBottomLeftInterpOverlap(
                    interp[0] * std::max<int >(0, x-blockOverlap), 
                    interp[1] * std::max<int >(0, y-blockOverlap), 
                    interp[2] * std::max<int >(0, z-blockOverlap));
                
				Shape coordTopRightInterp(
					interp[0] * std::min<int >(x+blockSizeRaw[0], sizeRaw[0]), 
					interp[1] * std::min<int >(y+blockSizeRaw[1], sizeRaw[1]), 
					interp[2] * std::min<int >(z+blockSizeRaw[2], sizeRaw[2]));
				Shape coordTopRightInterpOverlap(
					interp[0] * std::min<int >(x+blockSizeRaw[0]+blockOverlap, sizeRaw[0]), 
					interp[1] * std::min<int >(y+blockSizeRaw[1]+blockOverlap, sizeRaw[1]), 
					interp[2] * std::min<int >(z+blockSizeRaw[2]+blockOverlap, sizeRaw[2]));
                
                Shape coordBottomLeftDiff = coordBottomLeftInterp - coordBottomLeftInterpOverlap;
                Shape coordTopRightDiff = coordBottomLeftDiff + coordTopRightInterp - coordBottomLeftInterp;
                
                //std::cerr << "*****" << std::endl;
                //std::cerr << coordBottomLeftInterp << " - " << coordTopRightInterp << std::endl;
                //std::cerr << coordBottomLeftInterpOverlap << " - " << coordTopRightInterpOverlap << std::endl;
                //std::cerr << coordBottomLeftDiff << " - " << coordTopRightDiff << std::endl;
                //std::cerr << "*****" << std::endl;
                
                // read tmp data
                sprintf(buf, "/tmp/%04d_%d_%d_%d.h5", frame, i, j, k);
                MultiArray<DIM, TIN > data;
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    Shape shape = hdf5GetDatasetSize<DIM, TIN >(buf, "/", strVariable);
                    data.reshape(shape);
				    hdf5Read<DIM, TIN >(data, buf, "/", strVariable);
                }
                //std::cerr << "read in tmp file: " << buf << "; shape = " << data.shape() << std::endl;

                // do interpolation
                MultiArray<DIM, float > dataInterp;
                vigraResize<DIM, TIN, float >(data, interp, dataInterp);
                //std::cerr << "\t\tresizing completed" << std::endl;

                // filtering raw data
                MultiArray<DIM, TIN > dataFiltered;
                vigraIntensityThresholding<DIM, float, TIN >(dataInterp, intensityThreshold, dataFiltered);
                //std::cerr << "\t\tintensity thresholding completed" << std::endl;
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    vigra::HDF5File hdf5file(outFile, true);
                    hdf5file.cd(groupRaw);
                    MultiArray<DIM, TIN > tmp(coordTopRightDiff - coordBottomLeftDiff);
                    tmp.copy(dataFiltered.subarray(coordBottomLeftDiff, coordTopRightDiff));
                    hdf5file.write_block(varRaw, coordBottomLeftInterp, tmp);
                    //std::cerr << "cpu " << cpuId << ": raw data filtering completed -> " << buf << std::endl;
                }

                // segmentation
                MultiArray<DIM, TOUT > seg(dataInterp.shape(), static_cast<TOUT >(0));
                segMSA.run<DIM, float, TOUT >(dataInterp, seg);
                //std::cerr << "\t\tsegmentation completed" << std::endl;
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    vigra::HDF5File hdf5file(outFile, true);
                    hdf5file.cd(groupSegmentation);
                    MultiArray<DIM, TOUT > tmp(coordTopRightDiff - coordBottomLeftDiff);
                    tmp.copy(seg.subarray(coordBottomLeftDiff, coordTopRightDiff));
                    hdf5file.write_block(varSegmentation, coordBottomLeftInterp, tmp);
                    //std::cerr << "cpu " << cpuId << ": segmentation completed -> " << buf << std::endl;
                }
                
                // clean up
                {
                    boost::mutex::scoped_lock lock(io_mutex);
                    if( remove( buf ) != 0 )
                        std::cerr << "cpu " << cpuId << ": warning - failed to delete the tmp file -> " << buf << std::endl;
                    else
                        std::cerr << "cpu " << cpuId << ": successfully deleted the tmp file -> " << buf << std::endl;
                }
            }
        }
    }
}

template<int DIM, class TIN, class TOUT >
void segmentation(int frame, std::string section, std::string inifile)
{
    typedef typename MultiArray<DIM, TIN >::difference_type Shape;
    // load ini file
    char buf[1024];
    CSimpleIniA ini;
    ini.SetUnicode();
    ini.LoadFile(inifile.c_str());
    
    // split data
    split_export<DIM, TIN, TOUT >(frame, section, inifile);
    //std::cerr << "\t\tsplit_export completed for frame: " << frame << std::endl;
    
    //interpolate_segment_filter
    interpolate_segment_filter<DIM, TIN, TOUT >(frame, section, inifile);
    //std::cerr << "\t\tinterpolate_segment_filter completed for frame: " << frame << std::endl;
}

#endif /* __CT_SEGMENTATION_ROUTINE__ */