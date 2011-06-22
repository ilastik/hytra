#ifndef __HDF5_FILE_READER_WRITER__
#define __HDF5_FILE_READER_WRITER__

#include <iostream>
#include "hdf5impex.hxx"
#include <vigra/multi_array.hxx>

using namespace vigra;

template<int DIM, class T >
void hdf5Write(
    MultiArrayView<DIM, T > data,
    std::string infile, 
    std::string group, 
    std::string variable, 
    int chunk = 0, 
    int compression = 0)
{
    vigra::HDF5File hdf5file (infile, true);
    hdf5file.cd_mk(group);
    
    if (chunk > 0)
        hdf5file.enableChunks(chunk);
    
    if (compression > 0)
        hdf5file.enableCompression(compression);
    
    hdf5file.createDataset<DIM, T >(variable, data.shape(), static_cast<T >(0));
				
    typename MultiArrayView<DIM, T >::difference_type block_offset(0, 0, 0);
    hdf5file.write<DIM, T >(variable, data);
}


template<int DIM, class T >
void hdf5Read(
    MultiArrayView<DIM, T > data,
    std::string infile, 
    std::string group, 
    std::string variable)
{
    vigra::HDF5File hdf5file (infile, true);
    hdf5file.cd(group);
				
    hdf5file.read<DIM, T >(variable, data);
}

template<int DIM, class T >
typename MultiArrayView<DIM, T >::difference_type hdf5GetDatasetSize(
    std::string infile, 
    std::string group, 
    std::string variable)
{
    vigra::HDF5File hdf5file (infile, true);
    hdf5file.cd(group);
    
	hssize_t num_dimensions;
	vigra::ArrayVector<hsize_t> dim_shape;
    hdf5file.get_dataset_size(variable, num_dimensions, dim_shape);

    typename MultiArrayView<DIM, T >::difference_type shape;
    for (int i=0; i<DIM; i++)
        shape[i] = dim_shape[i];
    
    return shape;
}

#endif /* __HDF5_FILE_READER_WRITER__ */
