/************************************************************************/
/*                                                                      */
/*       Copyright 2009 by Michael Hanselmann and Ullrich Koethe        */
/*       Cognitive Systems Group, University of Hamburg, Germany        */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://kogs-www.informatik.uni-hamburg.de/~koethe/vigra/      */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#ifndef VIGRA_HDF5IMPEX_HXX
#define VIGRA_HDF5IMPEX_HXX

#include <string>
#include <hdf5.h>

#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR <= 6)
# ifndef H5Gopen
#   define H5Gopen(a, b, c) H5Gopen(a, b)
# endif
# ifndef H5Gcreate
#  define H5Gcreate(a, b, c, d, e) H5Gcreate(a, b, 1)
# endif
# ifndef H5Dopen
#  define H5Dopen(a, b, c) H5Dopen(a, b)
# endif
# ifndef H5Dcreate
#  define H5Dcreate(a, b, c, d, e, f, g) H5Dcreate(a, b, c, d, f)
# endif
# ifndef H5Acreate
#  define H5Acreate(a, b, c, d, e, f) H5Acreate(a, b, c, d, e)
# endif
# include <H5LT.h>
#else
# include <hdf5_hl.h>
#endif

#include "vigra/impex.hxx"
#include "vigra/multi_array.hxx"
#include "vigra/multi_impex.hxx"

namespace vigra {

class HDF5Handle
{
public:
    typedef herr_t (*Destructor)(hid_t);

private:
    hid_t handle_;
    Destructor destructor_;

public:

    HDF5Handle()
    : handle_( 0 ),
      destructor_(0)
    {}

    HDF5Handle(hid_t h, Destructor destructor, const char * error_message)
    : handle_( h ),
      destructor_(destructor)
    {
        if(handle_ < 0)
            vigra_fail(error_message);
    }

    HDF5Handle(HDF5Handle const & h)
    : handle_( h.handle_ ),
      destructor_(h.destructor_)
    {
        const_cast<HDF5Handle &>(h).handle_ = 0;
    }

    HDF5Handle & operator=(HDF5Handle const & h)
    {
        if(h.handle_ != handle_)
        {
            close();
            handle_ = h.handle_;
            destructor_ = h.destructor_;
            const_cast<HDF5Handle &>(h).handle_ = 0;
        }
        return *this;
    }

    ~HDF5Handle()
    {
        close();
    }
    
    herr_t close()
    {
        herr_t res = 1;
        if(handle_ && destructor_)
            res = (*destructor_)(handle_);
        handle_ = 0;
        return res;
    }

    hid_t get() const
    {
        return handle_;
    }

    operator hid_t() const
    {
        return handle_;
    }

    bool operator==(HDF5Handle const & h) const
    {
        return handle_ == h.handle_;
    }

    bool operator==(hid_t h) const
    {
        return handle_ == h;
    }

    bool operator!=(HDF5Handle const & h) const
    {
        return handle_ != h.handle_;
    }

    bool operator!=(hid_t h) const
    {
        return handle_ != h;
    }
};

/********************************************************/
/*                                                      */
/*                   HDF5ImportInfo                     */
/*                                                      */
/********************************************************/

/**
Namespace: vigra
**/
VIGRA_EXPORT class HDF5ImportInfo
{
  public:
    enum PixelType { UINT8, UINT16, UINT32, UINT64,
	   				 INT8, INT16, INT32, INT64,
					 FLOAT, DOUBLE };

        /** Construct HDF5ImageImportInfo object.

            The dataset in the given HDF5 file is accessed and the properties
            are set accordingly.
         **/
    VIGRA_EXPORT HDF5ImportInfo( const char* filePath, const char* pathInFile )
    {
        m_file_handle = HDF5Handle(H5Fopen(filePath, H5F_ACC_RDONLY, H5P_DEFAULT),
                                   &H5Fclose, "HDF5ImportInfo(): Unable to open file.");


        m_dataset_handle = HDF5Handle(H5Dopen(m_file_handle, pathInFile, H5P_DEFAULT),
                                      &H5Dclose, "HDF5ImportInfo(): Unable to open dataset.");


        //DataSet dset = m_file.openDataSet(datasetname);
        m_filename = filePath;
        m_path = pathInFile;
        HDF5Handle dataspace_handle(H5Dget_space(m_dataset_handle),
                                    &H5Sclose, "HDF5ImportInfo(): could not access dataset dataspace.");
        m_dimensions = H5Sget_simple_extent_ndims(dataspace_handle);
        //m_dimensions = dset.getSpace().getSimpleExtentNdims();

        //why?
        //vigra_precondition( m_dimensions>=2, "HDF5ImportInfo(): Number of dimensions is lower than 2. Not an image!" );

        hid_t datatype = H5Dget_type(m_dataset_handle);
        H5T_class_t dataclass = H5Tget_class(datatype);
        size_t datasize  = H5Tget_size(datatype);
        H5T_sign_t datasign  = H5Tget_sign(datatype);

        if(dataclass == H5T_FLOAT)
        {
            if(datasize == 4)
                m_pixeltype = "FLOAT";
            else if(datasize == 8)
                m_pixeltype = "DOUBLE";
        }
        else if(dataclass == H5T_INTEGER)
        {
            if(datasign == H5T_SGN_NONE)
            {
                if(datasize ==  1)
                    m_pixeltype = "UINT8";
                else if(datasize == 2)
                    m_pixeltype = "UINT16";
                else if(datasize == 4)
                    m_pixeltype = "UINT32";
                else if(datasize == 8)
                    m_pixeltype = "UINT64";
            }
            else
            {
                if(datasize ==  1)
                    m_pixeltype = "INT8";
                else if(datasize == 2)
                    m_pixeltype = "INT16";
                else if(datasize == 4)
                    m_pixeltype = "INT32";
                else if(datasize == 8)
                    m_pixeltype = "INT64";
            }
        }

        ArrayVector<hsize_t>::size_type ndims = ArrayVector<hsize_t>::size_type(m_dimensions);
        m_dims.resize(ndims);
        ArrayVector<hsize_t> size(ndims);
        ArrayVector<hsize_t> maxdims(ndims);
        H5Sget_simple_extent_dims(dataspace_handle, size.data(), maxdims.data());
        //dset.getSpace().getSimpleExtentDims(size, NULL);
        // invert the dimensions to guarantee c-order
        for(ArrayVector<hsize_t>::size_type i=0; i<ndims; i++) {
            m_dims[i] = size[ndims-1-i];
            //std::cout << "m_dims[" << i << "]=" << m_dims[i] << std::endl;
        }
    }

    VIGRA_EXPORT ~HDF5ImportInfo(){};

    VIGRA_EXPORT const std::string& getFilePath() const
            { return m_filename; }

    VIGRA_EXPORT const std::string& getPathInFile() const
            { return m_path; }

    VIGRA_EXPORT hid_t getH5FileHandle() const
            { return m_file_handle; }

    VIGRA_EXPORT hid_t getDatasetHandle() const
            { return m_dataset_handle; }

    VIGRA_EXPORT MultiArrayIndex numDimensions() const
            { return MultiArrayIndex(m_dimensions); }

    VIGRA_EXPORT ArrayVector<hsize_t> const & shape() const
    {
        return m_dims;
    }

    VIGRA_EXPORT MultiArrayIndex shapeOfDimension(const int dim) const
            { return MultiArrayIndex(m_dims[dim]); }

        /** Query the pixel type of the dataset.

            Possible values are:
            <DL>
            <DT>"UINT8"<DD> 8-bit unsigned integer (unsigned char)
            <DT>"INT16"<DD> 16-bit signed integer (short)
            <DT>"UINT16"<DD> 16-bit unsigned integer (unsigned short)
            <DT>"INT32"<DD> 32-bit signed integer (long)
            <DT>"UINT32"<DD> 32-bit unsigned integer (unsigned long)
            <DT>"FLOAT"<DD> 32-bit floating point (float)
            <DT>"DOUBLE"<DD> 64-bit floating point (double)
            </DL>
         **/
    VIGRA_EXPORT const char * getPixelType() const
    {
        return m_pixeltype.c_str();
    }

        /** Query the pixel type of the dataset.

            Same as getPixelType(), but the result is returned as a
            ImageImportInfo::PixelType enum. This is useful to implement
            a switch() on the pixel type.

            Possible values are:
            <DL>
            <DT>UINT8<DD> 8-bit unsigned integer (unsigned char)
            <DT>INT16<DD> 16-bit signed integer (short)
            <DT>UINT16<DD> 16-bit unsigned integer (unsigned short)
            <DT>INT32<DD> 32-bit signed integer (long)
            <DT>UINT32<DD> 32-bit unsigned integer (unsigned long)
            <DT>FLOAT<DD> 32-bit floating point (float)
            <DT>DOUBLE<DD> 64-bit floating point (double)
            </DL>
         **/
    VIGRA_EXPORT PixelType pixelType() const
    {
       const std::string pixeltype=HDF5ImportInfo::getPixelType();
       if (pixeltype == "UINT8")
           return HDF5ImportInfo::UINT8;
       if (pixeltype == "UINT16")
         return HDF5ImportInfo::UINT16;
       if (pixeltype == "UINT32")
         return HDF5ImportInfo::UINT32;
       if (pixeltype == "UINT64")
         return HDF5ImportInfo::UINT64;
       if (pixeltype == "INT8")
           return HDF5ImportInfo::INT8;
       if (pixeltype == "INT16")
         return HDF5ImportInfo::INT16;
       if (pixeltype == "INT32")
         return HDF5ImportInfo::INT32;
       if (pixeltype == "INT64")
         return HDF5ImportInfo::INT64;
       if (pixeltype == "FLOAT")
         return HDF5ImportInfo::FLOAT;
       if (pixeltype == "DOUBLE")
         return HDF5ImportInfo::DOUBLE;
       vigra_fail( "internal error: unknown pixel type" );
       return HDF5ImportInfo::PixelType();
    }

  private:
    HDF5Handle m_file_handle, m_dataset_handle;
    std::string m_filename, m_path, m_pixeltype;
    hssize_t m_dimensions;
    ArrayVector<hsize_t> m_dims;
};

namespace detail {

template<class type>
inline hid_t getH5DataType()
{
	std::runtime_error("getH5DataType(): invalid type");
	return 0;
}

#define VIGRA_H5_DATATYPE(type, h5type) \
template<> \
inline hid_t getH5DataType<type>() \
{ return h5type;}
VIGRA_H5_DATATYPE(char, H5T_NATIVE_CHAR)
VIGRA_H5_DATATYPE(Int8, H5T_NATIVE_INT8)
VIGRA_H5_DATATYPE(Int16, H5T_NATIVE_INT16)
VIGRA_H5_DATATYPE(Int32, H5T_NATIVE_INT32)
VIGRA_H5_DATATYPE(Int64, H5T_NATIVE_INT64)
VIGRA_H5_DATATYPE(UInt8, H5T_NATIVE_UINT8)
VIGRA_H5_DATATYPE(UInt16, H5T_NATIVE_UINT16)
VIGRA_H5_DATATYPE(UInt32, H5T_NATIVE_UINT32)
VIGRA_H5_DATATYPE(UInt64, H5T_NATIVE_UINT64)
VIGRA_H5_DATATYPE(float, H5T_NATIVE_FLOAT)
VIGRA_H5_DATATYPE(double, H5T_NATIVE_DOUBLE)
VIGRA_H5_DATATYPE(long double, H5T_NATIVE_LDOUBLE)

#undef VIGRA_H5_DATATYPE

} // namespace detail

/********************************************************/
/*                                                      */
/*                     HDF5File                         */
/*                                                      */
/********************************************************/

class HDF5File
{
  private:
	HDF5Handle file_handle_;
	HDF5Handle current_group_handle_;

    bool chunking_;
    int chunkSize_; // only support cubic chunks

    bool compression_;
    int compressionParameter_;


  public:
    /** Construct HDF5File object.
     * Create the file, if nescessary. Otherwise open it (append = true) or recreate it.
     * Initial group will be the root group "/".
     */
    HDF5File(std::string filename, bool append)
    {
        std::string message = "HDF5File: Could not create file '" + filename + "'.";
        file_handle_ = HDF5Handle(createFile_(filename, append),&H5Fclose,message.c_str());

        message = "HDF5File: Could not open group '/'.";
        current_group_handle_ = HDF5Handle(H5Gopen(file_handle_, "/", H5P_DEFAULT),&H5Gclose,message.c_str());
        chunking_ = false;
        compression_ = false;

    }


    HDF5File(std::string filename, std::string group, bool append)
    {
        std::string message = "HDF5File: Could not create file '" + filename + "'.";
        file_handle_ = HDF5Handle(createFile_(filename, append),&H5Fclose,message.c_str());

        message = "HDF5File: Could not open group '" + group + "'.";
        current_group_handle_ = HDF5Handle(createGroup_(file_handle_,group),&H5Gclose,message.c_str());

        chunking_ = false;
        compression_ = false;
    }


	/** Destructor to make sure that all data is flushed before closing the file.
     * Create the file, if nescessary. Otherwise open it (append = true) or recreate it.
     * The group "group" will be opened. If nescessary, it will be created before.
	 */
    ~HDF5File()
    {
        //Write everything to disk before closing
        H5Fflush(file_handle_, H5F_SCOPE_GLOBAL);
    }

    void enableChunks(int chunkSize)
    {
        chunkSize_ = chunkSize;
        chunking_ = true;
    }

    void disableChunks()
    {
        chunking_ = false;
    }

    void enableCompression(int compression)
    {
        compressionParameter_ = compression;
        compression_ = true;
    }

    void disableCompression()
    {
        compression_ = false;
    }

	/** current group to "/"
	 */
    void root()
    {
        std::string message = "HDF5File::root(): Could not open group '/'.";
        current_group_handle_ = HDF5Handle(H5Gopen(file_handle_, "/", H5P_DEFAULT),&H5Gclose,message.c_str());
    }


	/** change current group
     * If the first character is a "/", then the path will be interpreted as absolute path,
	 * otherwise it will be interpreted as a relative path to the current group.
	 */
    void cd(std::string group_name)
    {
        if(group_name ==".."){
            cd_up();
        }else{
            std::string message = "HDF5File::cd(): Could not open group '" + group_name + "'.";
            current_group_handle_ = HDF5Handle(openGroup_(current_group_handle_, group_name),&H5Gclose,message.c_str());
        }
    }

	/** change current group to parent group
	 */
    void cd_up()
    {
        std::string group_name = current_group_name_();

        //do not try to move up if we already in "/"
        if(group_name == "/"){
            std::cerr << "HDF5File::cd_up(): Could not move up one group. Already reached root group.\n";
            return;
        }

        size_t last_slash = group_name.find_last_of('/');

        std::string parent_group (group_name.begin(), group_name.begin()+last_slash+1);

        cd(parent_group);
    }

    void cd_up(int levels)
    {
        for(int i = 0; i<levels; i++){
            cd_up();
        }
    }


	/** create a group called "group_name" in the current group
	 * If the first character is a "/", then the path will be interpreted as a absolute path,
	 * otherwise it will be interpreted as a relative path to the current group.
	 */
    void mkdir(std::string group_name)
    {
        std::string message = "HDF5File::mkdir(): Could not create group '" + group_name + "'.";
        hid_t handle = createGroup_(current_group_handle_, group_name.c_str());
        if (handle != current_group_handle_){
            H5Gclose(handle);
        }
    }

	/** change current group. create it, if it does not exist yet
	 * If the first character is a "/", then the path will be interpreted as a absolute path,
	 * otherwise it will be interpreted as a relative path to the current group.
	 */
    void cd_mk(std::string group_name)
    {
        std::string message = "HDF5File::cd_mk(): Could not create group '" + group_name + "'.";
        current_group_handle_ = HDF5Handle(createGroup_(current_group_handle_, group_name.c_str()),&H5Gclose,message.c_str());
    }

	/** return the current group name
	 */
    std::string pwd()
    {
        return current_group_name_();
    }

	/** return the file name
	 */
    std::string filename()
    {
        return file_name_();
    }

	/** Get the number of dimensions and the dimension's shape of dataset dataset_name
	 *
	 * This function is essentially the same as get_dataset_info_, only without returned handles
	 */
    void get_dataset_size(std::string dataset_name, hssize_t &dimensions, ArrayVector<hsize_t> &dim_shape)
    {
        HDF5Handle temp;
        get_dataset_info_(dataset_name, temp, dimensions, dim_shape);
    }

	/** Attach an attribute to an existing object.
	 *
	 * The attribute can be attached to datasets.
	 * For simplicity, we will only use attributes of type string, which will be attached as a "label".
	 */
    void set_attribute(std::string dataset_name, std::string text)
    {
        std::string group_name;
        std::string::size_type delimiter = dataset_name.rfind('/');

        if(delimiter != std::string::npos)
        {
            group_name = std::string(dataset_name.begin(), dataset_name.begin()+delimiter);
            dataset_name = std::string(dataset_name.begin()+delimiter+1, dataset_name.end());
        }

        if(group_name != ""){
            HDF5Handle group_handle (openGroup_(current_group_handle_,group_name.c_str()),&H5Gclose,"set_attribute(): Could not open group.");
            H5LTset_attribute_string(group_handle,dataset_name.c_str(),"label",text.c_str());
        }else{
            H5LTset_attribute_string(current_group_handle_,dataset_name.c_str(),"label",text.c_str());
        }
    }

	/** Get the value of an attribute.
	 *
	 * parent_object must be a group or dataset name.
	 * For simplicity, we will only use attributes of type string, which are attached as a "label".
	 */
    std::string get_attribute(std::string dataset_name)
    {
        std::string group_name;
        std::string::size_type delimiter = dataset_name.rfind('/');

        if(delimiter != std::string::npos)
        {
            group_name = std::string(dataset_name.begin(), dataset_name.begin()+delimiter);
            dataset_name = std::string(dataset_name.begin()+delimiter+1, dataset_name.end());
        }

        HDF5Handle group_handle (openGroup_(current_group_handle_,group_name.c_str()),&H5Gclose,"set_attribute(): Could not open group.");

        char text[1000];
        if(group_name != ""){
            HDF5Handle group_handle (openGroup_(current_group_handle_,group_name.c_str()),&H5Gclose,"set_attribute(): Could not open group.");
            H5LTget_attribute_string(group_handle,dataset_name.c_str(),"label",text);
        }else{
            H5LTget_attribute_string(current_group_handle_,dataset_name.c_str(),"label",text);
        }

        return std::string(text);
    }


    /** Reading and writing data.

     **/

	// scalar and unstrided multi arrays
	template<unsigned int N, class T>
	inline void write(std::string dataset_name, const MultiArrayView<N, T, UnstridedArrayTag> & array) // scalar
	{
		write(dataset_name, array, detail::getH5DataType<T>(), 1);
	}

	template<unsigned int N, class T>
	inline void write_block(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, const MultiArrayView<N, T, UnstridedArrayTag> & array) // scalar
	{
		write_block(dataset_name, block_offset, array, detail::getH5DataType<T>(), 1);
	}

	// non-scalar (TinyVector) and unstrided multi arrays
	template<unsigned int N, class T, int SIZE>
	inline void write(std::string dataset_name, const MultiArrayView<N, TinyVector<T, SIZE>, UnstridedArrayTag> & array)
	{
		write(dataset_name, array, detail::getH5DataType<T>(), SIZE);
	}

	// non-scalar (RGBValue) and unstrided multi arrays
	template<unsigned int N, class T>
	inline void write(std::string dataset_name, const MultiArrayView<N, RGBValue<T>, UnstridedArrayTag> & array)
	{
		write(dataset_name, array, detail::getH5DataType<T>(), 3);
	}

	// unstrided multi arrays
	template<unsigned int N, class T>
	void write(std::string dataset_name, const MultiArrayView<N, T, UnstridedArrayTag> & array, const hid_t datatype, const int numBandsOfType)
	{
		HDF5Handle dataset_handle;
		createDataset_(dataset_name, array, datatype, numBandsOfType, dataset_handle);

	    // Write the data to the HDF5 dataset as is
		H5Dwrite( dataset_handle, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data()); // .data() possible since void pointer!
	}

	template<unsigned int N, class T>
	void write_block(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, const MultiArrayView<N, T, UnstridedArrayTag> & array, const hid_t datatype, const int numBandsOfType)
	{
		// open dataset if it exists
		std::string groupname;
		std::string setname;
		if(dataset_name.rfind("/") != std::string::npos){
			groupname = std::string(dataset_name.begin(),dataset_name.begin()+dataset_name.rfind("/"));
			setname = std::string(dataset_name.begin()+dataset_name.rfind("/")+1,dataset_name.end());
		}else{
			setname = dataset_name;
		}

		hid_t parent;
		bool close_parent = false;
		if(groupname.size()!=0){
			parent = createGroup_(current_group_handle_,groupname);
			close_parent  = true;
		}else{
			parent = current_group_handle_;
			close_parent = false;
		}

		HDF5Handle dataset_handle;

		hid_t id = H5Dopen(parent,setname.c_str(),H5P_DEFAULT);
		if(id > 0){
			dataset_handle = HDF5Handle(id,&H5Dclose,"Could not open dataset.");
		}else{
			std::cerr << "Error: No dataset available. Create target dataset first.";
			return;
		}

		// check if size fits
		//array.size(0) + block_offset[0]


		// hyperslab parameters for position, size, ...
		hsize_t boffset [N];
		hsize_t bshape [N];
		hsize_t bones [N];

		for(int i = 0; i < N; i++){
			boffset[i] = block_offset[N-1-i];
			bshape[i] = array.size(N-1-i);
			bones[i] = 1;
		}

		// create a target dataspace in memory with the shape of the desired block
		HDF5Handle memspace_handle (H5Screate_simple(N,bshape,NULL),&H5Sclose,"Unable to get origin dataspace");

		// get file dataspace and select the desired block
		HDF5Handle dataspace_handle (H5Dget_space(dataset_handle),&H5Sclose,"Unable to create target dataspace");
		H5Sselect_hyperslab(dataspace_handle, H5S_SELECT_SET, boffset, bones, bones, bshape);




	    // Write the data to the HDF5 dataset as is
		H5Dwrite( dataset_handle, datatype, memspace_handle, dataspace_handle, H5P_DEFAULT, array.data()); // .data() possible since void pointer!

	    //close group if we opened a different group than current_group
	    if(close_parent){
	    	H5Gclose(parent);
	    }
	}

	// scalar and strided multi arrays
	template<unsigned int N, class T>
	inline void write(std::string dataset_name, const MultiArrayView<N, T, StridedArrayTag> & array) // scalar
	{
		write(dataset_name, array, detail::getH5DataType<T>(), 1);
	}

	// non-scalar (TinyVector) and strided multi arrays
	template<unsigned int N, class T, int SIZE>
	inline void write(std::string dataset_name, const MultiArrayView<N, TinyVector<T, SIZE>, StridedArrayTag> & array)
	{
		write(dataset_name, array, detail::getH5DataType<T>(), SIZE);
	}

	// non-scalar (RGBValue) and strided multi arrays
	template<unsigned int N, class T>
	inline void write(std::string dataset_name, const MultiArrayView<N, RGBValue<T>, StridedArrayTag> & array)
	{
		write(dataset_name, array, detail::getH5DataType<T>(), 3);
	}
	// strided multi arrays
	template<unsigned int N, class T>
	void write(std::string dataset_name, const MultiArrayView<N, T, StridedArrayTag> & array, const hid_t datatype, const int numBandsOfType)
	{
		HDF5Handle dataset_handle;

		createDataset_(dataset_name, array, datatype, numBandsOfType, dataset_handle);

	    vigra::TinyVector<int,N> shape;
	    vigra::TinyVector<int,N> stride;
	    int elements = numBandsOfType;
	    for(unsigned int k=0; k<N; ++k)
	    {
	        shape[k] = array.shape(k);
			stride[k] = array.stride(k);
	        elements *= (int)shape[k];
	    }
	    int counter = 0;

	    ArrayVector<T> buffer((int)array.shape(0));
		writeHDF5Impl_(array.traverser_begin(), shape, dataset_handle, datatype, buffer, counter, elements, numBandsOfType, vigra::MetaInt<N-1>());

	}

	// scalar and unstrided target multi array
	template<unsigned int N, class T>
	inline void read(std::string dataset_name, MultiArrayView<N, T, UnstridedArrayTag> array) // scalar
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), 1);
	}

	template<unsigned int N, class T>
	inline void read_block(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, typename MultiArrayShape<N>::type block_shape, MultiArrayView<N, T, UnstridedArrayTag> array) // scalar
	{
		readHDF5_block_(dataset_name, block_offset, block_shape, array, detail::getH5DataType<T>(), 1);
	}

	// non-scalar (TinyVector) and unstrided target multi array
	template<unsigned int N, class T, int SIZE>
	inline void read(std::string dataset_name, MultiArrayView<N, TinyVector<T, SIZE>, UnstridedArrayTag> array)
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), SIZE);
	}

	template<unsigned int N, class T, int SIZE>
	inline void read_block(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, typename MultiArrayShape<N>::type block_shape, MultiArrayView<N, TinyVector<T, SIZE>, UnstridedArrayTag> array)
	{
		readHDF5_block_(dataset_name, block_offset, block_shape, array, detail::getH5DataType<T>(), SIZE);
	}

	// non-scalar (RGBValue) and unstrided target multi array
	template<unsigned int N, class T>
	inline void read(std::string dataset_name, MultiArrayView<N, RGBValue<T>, UnstridedArrayTag> array)
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), 3);
	}

	template<unsigned int N, class T>
	inline void read_block(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, typename MultiArrayShape<N>::type block_shape, MultiArrayView<N, RGBValue<T>, UnstridedArrayTag> array)
	{
		readHDF5_block_(dataset_name, block_offset, block_shape, array, detail::getH5DataType<T>(), 3);
	}

	// scalar and strided target multi array
	template<unsigned int N, class T>
	inline void read(std::string dataset_name, MultiArrayView<N, T, StridedArrayTag> array) // scalar
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), 1);
	}

	// non-scalar (TinyVector) and strided target multi array
	template<unsigned int N, class T, int SIZE>
	inline void read(std::string dataset_name, MultiArrayView<N, TinyVector<T, SIZE>, StridedArrayTag> array)
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), SIZE);
	}

	// non-scalar (RGBValue) and strided target multi array
	template<unsigned int N, class T>
	inline void read(std::string dataset_name, MultiArrayView<N, RGBValue<T>, StridedArrayTag> array)
	{
		readHDF5_(dataset_name, array, detail::getH5DataType<T>(), 3);
	}

    // scalar only: create a new dataset initialized with init
	template<int N, class T>
	inline void createDataset(std::string dataset_name, TinyVector<MultiArrayIndex, N> shape, T init)
	{
		//int N = 3;
		std::string groupname;
		std::string setname;
		if(dataset_name.rfind("/") != std::string::npos){
			groupname = std::string(dataset_name.begin(),dataset_name.begin()+dataset_name.rfind("/"));
			setname = std::string(dataset_name.begin()+dataset_name.rfind("/")+1,dataset_name.end());
		}else{
			setname = dataset_name;
		}

		hid_t parent;
		bool close_parent = false;
		if(groupname.size()!=0){
			parent = createGroup_(current_group_handle_,groupname);
			close_parent  = true;
		}else{
			parent = current_group_handle_;
			close_parent = false;
		}

		// delete the dataset if it already exists
		deleteDataset_(parent, setname);

		// create dataspace
		// add an extra dimension in case that the data is non-scalar
		HDF5Handle dataspace_handle;
		// invert dimensions to guarantee c-order

		hsize_t shape_inv[N];
		for(unsigned int k=0; k<N; ++k)
			shape_inv[N-1-k] = shape[k];

		// create dataspace
		dataspace_handle = HDF5Handle(H5Screate_simple(N, shape_inv, NULL),
									&H5Sclose, "createDataset_(): unable to create dataspace for scalar data.");

		// set fill value
		HDF5Handle plist ( H5Pcreate(H5P_DATASET_CREATE), &H5Pclose, "createDataset: unable to create property list." );
		H5Pset_fill_value(plist,detail::getH5DataType<T>(), &init);

        // enable chunks
        if(chunking_)
        {
            hsize_t cSize [N];
            for(int i = 0; i<N; i++)
            {
                cSize[i] = chunkSize_;
            }
            H5Pset_chunk (plist, N, cSize);
        }

        // enable compression
        if(compression_)
        {
            H5Pset_deflate(plist, compressionParameter_);
        }

		//create the dataset.
		HDF5Handle dataset_handle ( H5Dcreate(parent, setname.c_str(), detail::getH5DataType<T>(), dataspace_handle, H5P_DEFAULT, plist, H5P_DEFAULT),
								  &H5Dclose, "createDataset_(): unable to create dataset.");
		//close group if we opened a different group than current_group
		if(close_parent){
			H5Gclose(parent);
		}
	}

	/** Immediately write all data to disk
	 *
	 */
    void flush_to_disk()
    {
        H5Fflush(file_handle_, H5F_SCOPE_GLOBAL);
    }


  private:

	std::string current_group_name_()
	{
		int len = H5Iget_name(current_group_handle_,NULL,1000);
		char name [len+1];
		H5Iget_name(current_group_handle_,name,len+1);

		return std::string(name);
	}

	std::string file_name_()
	{
		int len = H5Fget_name(file_handle_,NULL,1000);
		char name [len+1];
		H5Fget_name(file_handle_,name,len+1);

		return std::string(name);
	}

	inline hid_t openGroup_(hid_t parent, std::string group_name)
	{
		//check if we want to change to root group
		if(group_name == "/"){
			return H5Gopen(file_handle_, "/", H5P_DEFAULT);
		}
		//check for empty group name
		if(group_name == ""){
			//reopen current group. returns a group handle even if current_group_handle_ is a filehandle
			return H5Gopen(file_handle_, current_group_name_().c_str(), H5P_DEFAULT);
		}

		//automatically set parent to root, if a absolute path is provided. Then remove leading "/".
		if (*group_name.begin() == '/'){
			parent = file_handle_;
			group_name = std::string(group_name.begin()+1,group_name.end());
		}

		//open subgroups one by one
	    size_t last_slash = group_name.find_last_of('/');
	    if (last_slash == std::string::npos || last_slash != group_name.size() - 1)
	        group_name = group_name + '/';

	    std::string::size_type begin = 0, end = group_name.find('/');
	    int ii =  0;
	    while (end != std::string::npos)
	    {
	    	std::string group(group_name.begin()+begin, group_name.begin()+end);
	        hid_t prev_parent = parent;
	        parent = H5Gopen(prev_parent, group.c_str(), H5P_DEFAULT);

	        if(ii != 0)     H5Gclose(prev_parent);
	        if(parent < 0)  return parent;
	        ++ii;
	        begin = end + 1;
	        end = group_name.find('/', begin);
	    }
	    return parent;
	}

	inline hid_t createGroup_(hid_t parent, std::string group_name)
	{
	    if(group_name.size() == 0 ||*group_name.rbegin() != '/')
	        group_name = group_name + '/';
	    if(group_name == "/")
	        return H5Gopen(parent, group_name.c_str(), H5P_DEFAULT);

		//automatically set parent to root, if a absolute path is provided. Then remove leading "/".
		if (*group_name.begin() == '/'){
			parent = file_handle_;
			group_name = std::string(group_name.begin()+1,group_name.end());
		}

		//create subgroups one by one
	    std::string::size_type begin = 0, end = group_name.find('/');
	    int ii =  0;
	    while (end != std::string::npos)
	    {
	        std::string group(group_name.begin()+begin, group_name.begin()+end);
	        hid_t prev_parent = parent;

	        if(H5LTfind_dataset(parent, group.c_str()) == 0)
	        {
	            parent = H5Gcreate(prev_parent, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	        } else {
	            parent = H5Gopen(prev_parent, group.c_str(), H5P_DEFAULT);
	        }

	        if(ii != 0)     H5Gclose(prev_parent);
	        if(parent < 0)  return parent;
	        ++ii;
	        begin = end + 1;
	        end = group_name.find('/', begin);
	    }
	    return parent;
	}

	inline void deleteDataset_(hid_t parent, std::string dataset_name)
	{
	    // delete existing data and create new dataset
	    if(H5LTfind_dataset(parent, dataset_name.c_str()))
	    {
	        //std::cout << "dataset already exists" << std::endl;
	#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR <= 6)
			if(H5Gunlink(parent, dataset_name.c_str()) < 0)
	        {
	            vigra_postcondition(false, "writeToHDF5File(): Unable to delete existing data.");
	        }
	#else
			if(H5Ldelete(parent, dataset_name.c_str(), H5P_DEFAULT ) < 0)
	        {
	            vigra_postcondition(false, "createDataset(): Unable to delete existing data.");
	        }
	#endif
	    }
	}

	inline hid_t createFile_(std::string filePath, bool append_ = true)
	{
	    FILE * pFile;
	    pFile = fopen ( filePath.c_str(), "r" );
	    hid_t file_id;
	    if ( pFile == NULL )
	    {
	        file_id = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	    }
	    else if(append_)
	    {
	        fclose( pFile );
	        file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	    }
	    else
	    {
	        fclose(pFile);
	        std::remove(filePath.c_str());
	        file_id = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	    }
	    return file_id;
	}

	//create a dataset called dataset_name in current group or in root group
	template<unsigned int N, class T, class Tag>
	void createDataset_(std::string dataset_name, const MultiArrayView<N, T, Tag> & array, const hid_t datatype, const int numBandsOfType, HDF5Handle & dataset_handle)
	{
		std::string groupname;
		std::string setname;
		if(dataset_name.rfind("/") != std::string::npos){
			groupname = std::string(dataset_name.begin(),dataset_name.begin()+dataset_name.rfind("/"));
			setname = std::string(dataset_name.begin()+dataset_name.rfind("/")+1,dataset_name.end());
		}else{
			setname = dataset_name;
		}

		hid_t parent;
		bool close_parent = false;
		if(groupname.size()!=0){
			parent = createGroup_(current_group_handle_,groupname);
			close_parent  = true;
		}else{
			parent = current_group_handle_;
			close_parent = false;
		}

	    // delete the dataset if it already exists
	    deleteDataset_(parent, setname);

        // set dataset creation properties
        HDF5Handle properties_handle(H5Pcreate(H5P_DATASET_CREATE), &H5Pclose,
                                     "HDF5File::createDataset_: could not create properties list.");

        // enable chunks
        if(chunking_)
        {
            hsize_t cSize [N];
            for(int i = 0; i<N; i++)
            {
                cSize[i] = chunkSize_;
            }
            H5Pset_chunk (properties_handle, N, cSize);
        }

        // enable compression
        if(compression_)
        {
            H5Pset_deflate(properties_handle, compressionParameter_);
        }

	    // create dataspace
		// add an extra dimension in case that the data is non-scalar
		HDF5Handle dataspace_handle;
		if(numBandsOfType > 1) {
			// invert dimensions to guarantee c-order
			hsize_t shape_inv[N+1]; // one additional dimension for pixel type channel(s)
			for(unsigned int k=0; k<N; ++k) {
				shape_inv[N-1-k] = array.shape(k);  // the channels (eg of an RGB image) are represented by the first dimension (before inversion)
				//std::cout << shape_inv[N-k] << " (" << N << ")";
			}
			shape_inv[N] = numBandsOfType;

			// create dataspace
			dataspace_handle = HDF5Handle(H5Screate_simple(N+1, shape_inv, NULL),
										&H5Sclose, "createDataset_(): unable to create dataspace for non-scalar data.");
		} else {
			// invert dimensions to guarantee c-order
			hsize_t shape_inv[N];
			for(unsigned int k=0; k<N; ++k)
				shape_inv[N-1-k] = array.shape(k);

			// create dataspace
			dataspace_handle = HDF5Handle(H5Screate_simple(N, shape_inv, NULL),
										&H5Sclose, "createDataset_(): unable to create dataspace for scalar data.");
		}

	    //alloc memory for dataset.
        dataset_handle = HDF5Handle(H5Dcreate( parent, setname.c_str(), datatype,
                                            dataspace_handle, H5P_DEFAULT, properties_handle, H5P_DEFAULT),
	                              &H5Dclose, "createDataset_(): unable to create dataset.");
	    //close group if we opened a different group than current_group
	    if(close_parent){
	    	H5Gclose(parent);
	    }
	}

	template <class Shape>
	inline void
	selectHyperslabs_(HDF5Handle & mid1, HDF5Handle & mid2, Shape const & shape, int & counter, const int elements, const int numBandsOfType)
	{
	    // select hyperslab in HDF5 file
	    hsize_t shapeHDF5[2];
	    shapeHDF5[0] = 1;
	    shapeHDF5[1] = elements;
	    hsize_t startHDF5[2];
	    startHDF5[0] = 0;
	    startHDF5[1] = counter * numBandsOfType * shape[0]; // we have to reserve space for the pixel type channel(s)
	    hsize_t strideHDF5[2];
	    strideHDF5[0] = 1;
	    strideHDF5[1] = 1;
	    hsize_t countHDF5[2];
	    countHDF5[0] = 1;
	    countHDF5[1] = numBandsOfType * shape[0];
	    hsize_t blockHDF5[2];
	    blockHDF5[0] = 1;
	    blockHDF5[1] = 1;
	    mid1 = HDF5Handle(H5Screate_simple(2, shapeHDF5, NULL),
	                      &H5Sclose, "unable to create hyperslabs.");
	    H5Sselect_hyperslab(mid1, H5S_SELECT_SET, startHDF5, strideHDF5, countHDF5, blockHDF5);
	    // select hyperslab in input data object
	    hsize_t shapeData[2];
	    shapeData[0] = 1;
	    shapeData[1] = numBandsOfType * shape[0];
	    hsize_t startData[2];
	    startData[0] = 0;
	    startData[1] = 0;
	    hsize_t strideData[2];
	    strideData[0] = 1;
	    strideData[1] = 1;
	    hsize_t countData[2];
	    countData[0] = 1;
	    countData[1] = numBandsOfType * shape[0];
	    hsize_t blockData[2];
	    blockData[0] = 1;
	    blockData[1] = 1;
	    mid2 = HDF5Handle(H5Screate_simple(2, shapeData, NULL),
	                      &H5Sclose, "unable to create hyperslabs.");
	    H5Sselect_hyperslab(mid2, H5S_SELECT_SET, startData, strideData, countData, blockData);
	}

    void get_dataset_info_(std::string dataset_name, HDF5Handle &dataset_handle, hssize_t &dimensions, ArrayVector<hsize_t> &dim_shape)
    {
        //Prepare to read without using HDF5ImportInfo

        std::string group_name;
        std::string set (dataset_name);
        std::string::size_type delimiter = set.rfind('/');

        if(delimiter == std::string::npos)
        {
            dataset_name = set;
        }
        else
        {
            group_name = std::string(set.begin(), set.begin()+delimiter);
            dataset_name = std::string(set.begin()+delimiter+1, set.end());
        }


        //Open parent group to distinguish between relative and absolute paths
        HDF5Handle group_handle = HDF5Handle(openGroup_(current_group_handle_, group_name.c_str()), &H5Gclose, "readHDF5_(): Unable to open group.");

        //Open dataset and dataspace
        dataset_handle = HDF5Handle(H5Dopen(group_handle, dataset_name.c_str(), H5P_DEFAULT), &H5Dclose, "readHDF5_(): Unable to open dataset.");
        HDF5Handle dataspace_handle(H5Dget_space(dataset_handle), &H5Sclose, "readHDF5_(): could not access dataset dataspace.");

        //Get dimension information
        dimensions = H5Sget_simple_extent_ndims(dataspace_handle);
        ArrayVector<hsize_t>::size_type ndims = ArrayVector<hsize_t>::size_type(dimensions);
        dim_shape.resize(ndims);
        ArrayVector<hsize_t> size(ndims);
        ArrayVector<hsize_t> maxdims(ndims);
        H5Sget_simple_extent_dims(dataspace_handle, size.data(), maxdims.data());

        // invert the dimensions to guarantee c-order

        for(ArrayVector<hsize_t>::size_type i=0; i<ndims; i++) {
            dim_shape[i] = size[ndims-1-i];
        }

        return;
    }


	template <class DestIterator, class Shape, class T>
	inline void
	writeHDF5Impl_(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<0>)
	{
	    DestIterator dend = d + (typename DestIterator::difference_type)shape[0];
	    int k = 0;
		//std::cout << "new:" << std::endl;
		for(; d < dend; ++d, k++)
	    {
	        buffer[k] = *d;
	        //std::cout << buffer[k] << " ";
	    }
		//std::cout << std::endl;
	    HDF5Handle mid1, mid2;

	    // select hyperslabs
	    selectHyperslabs_(mid1, mid2, shape, counter, elements, numBandsOfType);

	    // write to hdf5
	    H5Dwrite(dataset_id, datatype, mid2, mid1, H5P_DEFAULT, buffer.data());
	    // increase counter
	    counter++;
	}

	template <class DestIterator, class Shape, class T, int N>
	void
	writeHDF5Impl_(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<N>)
	{
			DestIterator dend = d + (typename DestIterator::difference_type)shape[N];
			for(; d < dend; ++d)
			{
				writeHDF5Impl_(d.begin(), shape, dataset_id, datatype, buffer, counter, elements, numBandsOfType, MetaInt<N-1>());
			}
	}


	template <class DestIterator, class Shape, class T>
	inline void
	readHDF5Impl_(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<0>)
	{
	    HDF5Handle mid1, mid2;

	    // select hyperslabs
	    selectHyperslabs_(mid1, mid2, shape, counter, elements, numBandsOfType);

	    // read from hdf5
	    H5Dread(dataset_id, datatype, mid2, mid1, H5P_DEFAULT, buffer.data());

	    // increase counter
	    counter++;


		//std::cout << "numBandsOfType: " << numBandsOfType << std::endl;
	    DestIterator dend = d + shape[0];
	    int k = 0;
	    for(; d < dend; ++d, k++)
	    {
	        *d = buffer[k];
	        //std::cout << buffer[k] << "| ";
	    }

	}

	template <class DestIterator, class Shape, class T, int N>
	void
	readHDF5Impl_(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<N>)
	{
	    DestIterator dend = d + shape[N];
	    for(; d < dend; ++d)
	    {
	        readHDF5Impl_(d.begin(), shape, dataset_id, datatype, buffer, counter, elements, numBandsOfType, MetaInt<N-1>());
	    }
	}

	// unstrided target multi array
	template<unsigned int N, class T>
	void readHDF5_(std::string dataset_name, MultiArrayView<N, T, UnstridedArrayTag> array, const hid_t datatype, const int numBandsOfType)
	{
		//Prepare to read without using HDF5ImportInfo
		ArrayVector<hsize_t> dims;
		hssize_t dimensions;
		HDF5Handle dataset_handle;

		get_dataset_info_(dataset_name, dataset_handle, dimensions, dims);

		int offset = (numBandsOfType > 1);

		//std::cout << "offset: " << offset << ", N: " << N << ", dims: " << MultiArrayIndex(dimensions) << std::endl;
		vigra_precondition(( (N + offset ) ==  MultiArrayIndex(dimensions)), // the object in the HDF5 file may have one additional dimension which we then interpret as the pixel type bands
	        "readHDF5(): Array dimension disagrees with data dimension.");

	    typename MultiArrayShape<N>::type shape;
		for(int k=offset; k< MultiArrayIndex(dimensions); ++k) {
	        shape[k-offset] = MultiArrayIndex(dims[k]);
		}

		vigra_precondition(shape == array.shape(),
	         "readHDF5(): Array shape disagrees with HDF5ImportInfo.");

		// simply read in the data as is
		H5Dread( dataset_handle, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data() ); // .data() possible since void pointer!
	}


	// strided target multi array
	template<unsigned int N, class T>
	void readHDF5_(std::string dataset_name, MultiArrayView<N, T, StridedArrayTag> array, const hid_t datatype, const int numBandsOfType)
	{
		//Prepare to read without using HDF5ImportInfo
		ArrayVector<hsize_t> dims;
		hssize_t dimensions;
		HDF5Handle dataset_handle;

		get_dataset_info_(dataset_name, dataset_handle, dimensions, dims);

		int offset = (numBandsOfType > 1);

		//std::cout << "offset: " << offset << ", N: " << N << ", dims: " << MultiArrayIndex(dimensions) << std::endl;
		vigra_precondition(( (N + offset ) == MultiArrayIndex(dimensions)), // the object in the HDF5 file may have one additional dimension which we then interpret as the pixel type bands
	        "readHDF5(): Array dimension disagrees with HDF5ImportInfo.numDimensions().");

	    typename MultiArrayShape<N>::type shape;
		for(int k=offset; k<MultiArrayIndex(dimensions); ++k) {
	        shape[k-offset] = MultiArrayIndex(dims[k]);
		}

		vigra_precondition(shape == array.shape(),
	         "readHDF5(): Array shape disagrees with HDF5ImportInfo.");

	    //Get the data
	    int counter = 0;
	    int elements = numBandsOfType;
	    for(unsigned int i=0;i<N;++i)
	        elements *= shape[i];
	    ArrayVector<T> buffer(shape[0]);
	    readHDF5Impl_(array.traverser_begin(), shape, dataset_handle, datatype, buffer, counter, elements, numBandsOfType, vigra::MetaInt<N-1>());
	}

	// unstrided target multi array
	// read a block of a HDF5 dataset into a MultiArray
	template<unsigned int N, class T>
	void readHDF5_block_(std::string dataset_name, typename MultiArrayShape<N>::type block_offset, typename MultiArrayShape<N>::type block_shape, MultiArrayView<N, T, UnstridedArrayTag> array, const hid_t datatype, const int numBandsOfType)
	{
		//Prepare to read without using HDF5ImportInfo
		ArrayVector<hsize_t> dims;
		hssize_t dimensions;
		HDF5Handle dataset_handle;

		get_dataset_info_(dataset_name, dataset_handle, dimensions, dims);

		int offset = (numBandsOfType > 1);

		vigra_precondition(( (N + offset ) ==  MultiArrayIndex(dimensions)), // the object in the HDF5 file may have one additional dimension which we then interpret as the pixel type bands
	        "readHDF5_block(): Array dimension disagrees with data dimension.");

	    /*typename MultiArrayShape<N>::type shape;
		for(int k=offset; k< MultiArrayIndex(dimensions); ++k) {
	        shape[k-offset] = MultiArrayIndex(dims[k]);
		}*/

		vigra_precondition(block_shape == array.shape(),
	         "readHDF5_block(): Array shape disagrees with block size.");

		// hyperslab parameters for position, size, ...
		hsize_t boffset [N];
		hsize_t bshape [N];
		hsize_t bones [N];

		for(int i = 0; i < N; i++){
			// virgra and hdf5 use different indexing
			boffset[i] = block_offset[N-1-i];
			//bshape[i] = block_shape[i];
			bshape[i] = block_shape[N-1-i];
			//boffset[i] = block_offset[N-1-i];
			bones[i] = 1;
		}

		// create a target dataspace in memory with the shape of the desired block
		HDF5Handle memspace_handle (H5Screate_simple(N,bshape,NULL),&H5Sclose,"Unable to create target dataspace");

		// get file dataspace and select the desired block
		HDF5Handle dataspace_handle (H5Dget_space(dataset_handle),&H5Sclose,"Unable to get dataspace");
		H5Sselect_hyperslab(dataspace_handle, H5S_SELECT_SET, boffset, bones, bones, bshape);

		// now read the data
		H5Dread( dataset_handle, datatype, memspace_handle, dataspace_handle, H5P_DEFAULT, array.data() ); // .data() possible since void pointer!
	}

};  /* class HDF5File */








namespace detail {

template <class Shape>
inline void
selectHyperslabs(HDF5Handle & mid1, HDF5Handle & mid2, Shape const & shape, int & counter, const int elements, const int numBandsOfType)
{
    // select hyperslab in HDF5 file
    hsize_t shapeHDF5[2];
    shapeHDF5[0] = 1;
    shapeHDF5[1] = elements;
    hsize_t startHDF5[2];
    startHDF5[0] = 0;
    startHDF5[1] = counter * numBandsOfType * shape[0]; // we have to reserve space for the pixel type channel(s)
    hsize_t strideHDF5[2];
    strideHDF5[0] = 1;
    strideHDF5[1] = 1;
    hsize_t countHDF5[2];
    countHDF5[0] = 1;
    countHDF5[1] = numBandsOfType * shape[0];
    hsize_t blockHDF5[2];
    blockHDF5[0] = 1;
    blockHDF5[1] = 1;
    mid1 = HDF5Handle(H5Screate_simple(2, shapeHDF5, NULL),
                      &H5Sclose, "unable to create hyperslabs.");
    H5Sselect_hyperslab(mid1, H5S_SELECT_SET, startHDF5, strideHDF5, countHDF5, blockHDF5);
    // select hyperslab in input data object
    hsize_t shapeData[2];
    shapeData[0] = 1;
    shapeData[1] = numBandsOfType * shape[0];
    hsize_t startData[2];
    startData[0] = 0;
    startData[1] = 0;
    hsize_t strideData[2];
    strideData[0] = 1;
    strideData[1] = 1;
    hsize_t countData[2];
    countData[0] = 1;
    countData[1] = numBandsOfType * shape[0];
    hsize_t blockData[2];
    blockData[0] = 1;
    blockData[1] = 1;
    mid2 = HDF5Handle(H5Screate_simple(2, shapeData, NULL),
                      &H5Sclose, "unable to create hyperslabs.");
    H5Sselect_hyperslab(mid2, H5S_SELECT_SET, startData, strideData, countData, blockData);
}

template <class DestIterator, class Shape, class T>
inline void
readHDF5Impl(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<0>)
{
    HDF5Handle mid1, mid2;

    // select hyperslabs
    selectHyperslabs(mid1, mid2, shape, counter, elements, numBandsOfType);

    // read from hdf5
    H5Dread(dataset_id, datatype, mid2, mid1, H5P_DEFAULT, buffer.data());

    // increase counter
    counter++;


	//std::cout << "numBandsOfType: " << numBandsOfType << std::endl;
    DestIterator dend = d + shape[0];
    int k = 0;
    for(; d < dend; ++d, k++)
    {
        *d = buffer[k];
        //std::cout << buffer[k] << "| ";
    }

}

template <class DestIterator, class Shape, class T, int N>
void
readHDF5Impl(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<N>)
{
    DestIterator dend = d + shape[N];
    for(; d < dend; ++d)
    {
        readHDF5Impl(d.begin(), shape, dataset_id, datatype, buffer, counter, elements, numBandsOfType, MetaInt<N-1>());
    }
}

} // namespace detail

// scalar and unstrided target multi array
template<unsigned int N, class T>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, T, UnstridedArrayTag> array) // scalar
{
	readHDF5(info, array, detail::getH5DataType<T>(), 1);
}

// non-scalar (TinyVector) and unstrided target multi array
template<unsigned int N, class T, int SIZE>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, TinyVector<T, SIZE>, UnstridedArrayTag> array)
{
	readHDF5(info, array, detail::getH5DataType<T>(), SIZE);
}

// non-scalar (RGBValue) and unstrided target multi array
template<unsigned int N, class T>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, RGBValue<T>, UnstridedArrayTag> array)
{
	readHDF5(info, array, detail::getH5DataType<T>(), 3);
}

// unstrided target multi array
template<unsigned int N, class T>
void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, T, UnstridedArrayTag> array, const hid_t datatype, const int numBandsOfType)
{
	int offset = (numBandsOfType > 1);

	//std::cout << "offset: " << offset << ", N: " << N << ", dims: " << info.numDimensions() << std::endl;
	vigra_precondition(( (N + offset ) == info.numDimensions()), // the object in the HDF5 file may have one additional dimension which we then interpret as the pixel type bands
        "readHDF5(): Array dimension disagrees with HDF5ImportInfo.numDimensions().");

    typename MultiArrayShape<N>::type shape;
	for(int k=offset; k<info.numDimensions(); ++k) {
        shape[k-offset] = info.shapeOfDimension(k);
	}

	vigra_precondition(shape == array.shape(),
         "readHDF5(): Array shape disagrees with HDF5ImportInfo.");

	// simply read in the data as is
	H5Dread( info.getDatasetHandle(), datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data() ); // .data() possible since void pointer!
}

// scalar and strided target multi array
template<unsigned int N, class T>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, T, StridedArrayTag> array) // scalar
{
	readHDF5(info, array, detail::getH5DataType<T>(), 1);
}

// non-scalar (TinyVector) and strided target multi array
template<unsigned int N, class T, int SIZE>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, TinyVector<T, SIZE>, StridedArrayTag> array)
{
	readHDF5(info, array, detail::getH5DataType<T>(), SIZE);
}

// non-scalar (RGBValue) and strided target multi array
template<unsigned int N, class T>
inline void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, RGBValue<T>, StridedArrayTag> array)
{
	readHDF5(info, array, detail::getH5DataType<T>(), 3);
}

// strided target multi array
template<unsigned int N, class T>
void readHDF5(const HDF5ImportInfo &info, MultiArrayView<N, T, StridedArrayTag> array, const hid_t datatype, const int numBandsOfType)
{
	int offset = (numBandsOfType > 1);

	//std::cout << "offset: " << offset << ", N: " << N << ", dims: " << info.numDimensions() << std::endl;
	vigra_precondition(( (N + offset ) == info.numDimensions()), // the object in the HDF5 file may have one additional dimension which we then interpret as the pixel type bands
        "readHDF5(): Array dimension disagrees with HDF5ImportInfo.numDimensions().");

    typename MultiArrayShape<N>::type shape;
	for(int k=offset; k<info.numDimensions(); ++k) {
        shape[k-offset] = info.shapeOfDimension(k);
	}

	vigra_precondition(shape == array.shape(),
         "readHDF5(): Array shape disagrees with HDF5ImportInfo.");

    //Get the data
    int counter = 0;
    int elements = numBandsOfType;
    for(unsigned int i=0;i<N;++i)
        elements *= shape[i];
    ArrayVector<T> buffer(shape[0]);
    detail::readHDF5Impl(array.traverser_begin(), shape, info.getDatasetHandle(), datatype, buffer, counter, elements, numBandsOfType, vigra::MetaInt<N-1>());
}

inline hid_t openGroup(hid_t parent, std::string group_name)
{
    //std::cout << group_name << std::endl;
    size_t last_slash = group_name.find_last_of('/');
    if (last_slash == std::string::npos || last_slash != group_name.size() - 1)
        group_name = group_name + '/';
    std::string::size_type begin = 0, end = group_name.find('/');
    int ii =  0;
    while (end != std::string::npos)
    {
        std::string group(group_name.begin()+begin, group_name.begin()+end);
        hid_t prev_parent = parent;
        parent = H5Gopen(prev_parent, group.c_str(), H5P_DEFAULT);

        if(ii != 0)     H5Gclose(prev_parent);
        if(parent < 0)  return parent;
        ++ii;
        begin = end + 1;
        end = group_name.find('/', begin);
    }
    return parent;
}

inline hid_t createGroup(hid_t parent, std::string group_name)
{
    if(group_name.size() == 0 ||*group_name.rbegin() != '/')
        group_name = group_name + '/';
    if(group_name == "/")
        return H5Gopen(parent, group_name.c_str(), H5P_DEFAULT);

    std::string::size_type begin = 0, end = group_name.find('/');
    int ii =  0;
    while (end != std::string::npos)
    {
        std::string group(group_name.begin()+begin, group_name.begin()+end);
        hid_t prev_parent = parent;

        if(H5LTfind_dataset(parent, group.c_str()) == 0)
        {
            parent = H5Gcreate(prev_parent, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        } else {
            parent = H5Gopen(prev_parent, group.c_str(), H5P_DEFAULT);
        }

        if(ii != 0)     H5Gclose(prev_parent);
        if(parent < 0)  return parent;
        ++ii;
        begin = end + 1;
        end = group_name.find('/', begin);
    }
    return parent;
}

inline void deleteDataset(hid_t parent, std::string dataset_name)
{
    // delete existing data and create new dataset
    if(H5LTfind_dataset(parent, dataset_name.c_str()))
    {
        //std::cout << "dataset already exists" << std::endl;
#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR <= 6)
		if(H5Gunlink(parent, dataset_name.c_str()) < 0)
        {
            vigra_postcondition(false, "writeToHDF5File(): Unable to delete existing data.");
        }
#else
		if(H5Ldelete(parent, dataset_name.c_str(), H5P_DEFAULT ) < 0)
        {
            vigra_postcondition(false, "createDataset(): Unable to delete existing data.");
        }
#endif
    }
}

inline hid_t createFile(std::string filePath, bool append_ = true)
{
    FILE * pFile;
    pFile = fopen ( filePath.c_str(), "r" );
    hid_t file_id;
    if ( pFile == NULL )
    {
        file_id = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    else if(append_)
    {
        fclose( pFile );
        file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    }
    else
    {
        fclose(pFile);
        std::remove(filePath.c_str());
        file_id = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    return file_id;
}

template<unsigned int N, class T, class Tag>
void createDataset(const char* filePath, const char* pathInFile, const MultiArrayView<N, T, Tag> & array, const hid_t datatype, const int numBandsOfType, HDF5Handle & file_handle, HDF5Handle & dataset_handle)
{
    std::string path_name(pathInFile), group_name, data_set_name, message;
    std::string::size_type delimiter = path_name.rfind('/');

    //create or open file
    file_handle = HDF5Handle(createFile(filePath), &H5Fclose,
                       "createDataset(): unable to open output file.");

    // get the groupname and the filename
    if(delimiter == std::string::npos)
    {
        group_name    = "/";
        data_set_name = path_name;
    }
    else
    {
        group_name = std::string(path_name.begin(), path_name.begin()+delimiter);
        data_set_name = std::string(path_name.begin()+delimiter+1, path_name.end());
    }

    // create all groups
    HDF5Handle group(createGroup(file_handle, group_name), &H5Gclose,
                     "createDataset(): Unable to create and open group. generic v");

    // delete the dataset if it already exists
    deleteDataset(group, data_set_name);

    // create dataspace
	// add an extra dimension in case that the data is non-scalar
	HDF5Handle dataspace_handle;
	if(numBandsOfType > 1) {
		// invert dimensions to guarantee c-order
		hsize_t shape_inv[N+1]; // one additional dimension for pixel type channel(s)
		for(unsigned int k=0; k<N; ++k) {
			shape_inv[N-1-k] = array.shape(k);  // the channels (eg of an RGB image) are represented by the first dimension (before inversion)
			//std::cout << shape_inv[N-k] << " (" << N << ")";
		}
		shape_inv[N] = numBandsOfType;

		// create dataspace
		dataspace_handle = HDF5Handle(H5Screate_simple(N+1, shape_inv, NULL),
									&H5Sclose, "createDataset(): unable to create dataspace for non-scalar data.");
	} else {
		// invert dimensions to guarantee c-order
		hsize_t shape_inv[N];
		for(unsigned int k=0; k<N; ++k)
			shape_inv[N-1-k] = array.shape(k);

		// create dataspace
		dataspace_handle = HDF5Handle(H5Screate_simple(N, shape_inv, NULL),
									&H5Sclose, "createDataset(): unable to create dataspace for scalar data.");
	}

    //alloc memory for dataset.
    dataset_handle = HDF5Handle(H5Dcreate(group,
                                        data_set_name.c_str(),
                                        datatype,
                                        dataspace_handle,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                              &H5Dclose, "createDataset(): unable to create dataset.");
}



namespace detail {

template <class DestIterator, class Shape, class T>
inline void
writeHDF5Impl(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<0>)
{
    DestIterator dend = d + (typename DestIterator::difference_type)shape[0];
    int k = 0;
	//std::cout << "new:" << std::endl;
	for(; d < dend; ++d, k++)
    {
        buffer[k] = *d;
        //std::cout << buffer[k] << " ";
    }
	std::cout << std::endl;
    HDF5Handle mid1, mid2;

    // select hyperslabs
    selectHyperslabs(mid1, mid2, shape, counter, elements, numBandsOfType);

    // write to hdf5
    H5Dwrite(dataset_id, datatype, mid2, mid1, H5P_DEFAULT, buffer.data());
    // increase counter
    counter++;
}

template <class DestIterator, class Shape, class T, int N>
void
writeHDF5Impl(DestIterator d, Shape const & shape, const hid_t dataset_id, const hid_t datatype, ArrayVector<T> & buffer, int & counter, const int elements, const int numBandsOfType, MetaInt<N>)
{
		DestIterator dend = d + (typename DestIterator::difference_type)shape[N];
		for(; d < dend; ++d)
		{
			writeHDF5Impl(d.begin(), shape, dataset_id, datatype, buffer, counter, elements, numBandsOfType, MetaInt<N-1>());
		}
}

} // namespace detail

/** write a MultiArrayView to hdf5 file
 */
// scalar and unstrided multi arrays
template<unsigned int N, class T>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, T, UnstridedArrayTag> & array) // scalar
{
	HDF5File file (filePath, true);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// non-scalar (TinyVector) and unstrided multi arrays
template<unsigned int N, class T, int SIZE>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, TinyVector<T, SIZE>, UnstridedArrayTag> & array)
{
	HDF5File file (filePath, true);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// non-scalar (RGBValue) and unstrided multi arrays
template<unsigned int N, class T>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, RGBValue<T>, UnstridedArrayTag> & array)
{
	HDF5File file (filePath, true);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// unstrided multi arrays
template<unsigned int N, class T>
void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, T, UnstridedArrayTag> & array, const hid_t datatype, const int numBandsOfType)
{
	HDF5File file (filePath, true);
	file.write(pathInFile, array, datatype, numBandsOfType);
	file.flush_to_disk();
}


// scalar and strided multi arrays
template<unsigned int N, class T>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, T, StridedArrayTag> & array) // scalar
{
	HDF5File file (filePath, true);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// non-scalar (TinyVector) and strided multi arrays
template<unsigned int N, class T, int SIZE>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, TinyVector<T, SIZE>, StridedArrayTag> & array)
{
	HDF5File file (filePath, false);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// non-scalar (RGBValue) and strided multi arrays
template<unsigned int N, class T>
inline void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, RGBValue<T>, StridedArrayTag> & array)
{
	HDF5File file (filePath, false);
	file.write(pathInFile,array);
	file.flush_to_disk();
}

// strided multi arrays
template<unsigned int N, class T>
void writeHDF5(const char* filePath, const char* pathInFile, const MultiArrayView<N, T, StridedArrayTag> & array, const hid_t datatype, const int numBandsOfType)
{
	HDF5File file (filePath, true);
	file.write(pathInFile,array, datatype, numBandsOfType);
	file.flush_to_disk();
}

namespace detail
{
struct MaxSizeFnc
{
    size_t size;

    MaxSizeFnc()
    : size(0)
    {}

    void operator()(std::string const & in)
    {
        size = in.size() > size ?
                    in.size() :
                    size;
    }
};
} /* namespace detail */


#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR == 8)
/** write a numeric MultiArray as a attribute of location identifier loc
 * with name name
 */
template<size_t N, class T, class C>
void writeHDF5Attr(hid_t loc,
                   const char* name,
                   MultiArrayView<N, T, C> const & array)
{
    if(H5Aexists(loc, name) > 0)
        H5Adelete(loc, name);

    ArrayVector<hsize_t> shape(array.shape().begin(),
                               array.shape().end());
    HDF5Handle
        dataspace_handle(H5Screate_simple(N, shape.data(), NULL),
                         &H5Sclose,
                         "writeToHDF5File(): unable to create dataspace.");

    HDF5Handle attr(H5Acreate(loc,
                              name,
                              detail::getH5DataType<T>(),
                              dataspace_handle,
                              H5P_DEFAULT ,H5P_DEFAULT ),
                    &H5Aclose,
                    "writeHDF5Attr: unable to create Attribute");

    //copy data - since attributes are small - who cares!
    ArrayVector<T> buffer;
    for(int ii = 0; ii < array.size(); ++ii)
        buffer.push_back(array[ii]);
    H5Awrite(attr, detail::getH5DataType<T>(), buffer.data());
}



/** write a String MultiArray as a attribute of location identifier
 *  loc with name name
 */
template<size_t N, class C>
void writeHDF5Attr(hid_t loc,
                   const char* name,
                   MultiArrayView<N, std::string, C> const & array)
{
    if(H5Aexists(loc, name) > 0)
        H5Adelete(loc, name);

    ArrayVector<hsize_t> shape(array.shape().begin(),
                               array.shape().end());
    HDF5Handle
        dataspace_handle(H5Screate_simple(N, shape.data(), NULL),
                         &H5Sclose,
                         "writeToHDF5File(): unable to create dataspace.");

    HDF5Handle atype(H5Tcopy (H5T_C_S1),
                     &H5Tclose,
                     "writeToHDF5File(): unable to create type.");

    detail::MaxSizeFnc max_size;
    max_size = std::for_each(array.data(),array.data()+ array.size(), max_size);
    H5Tset_size (atype, max_size.size);

    HDF5Handle attr(H5Acreate(loc,
                              name,
                              atype,
                              dataspace_handle,
                              H5P_DEFAULT ,H5P_DEFAULT ),
                    &H5Aclose,
                    "writeHDF5Attr: unable to create Attribute");

    std::string buf ="";
    for(int ii = 0; ii < array.size(); ++ii)
    {
        buf = buf + array[ii]
                  + std::string(max_size.size - array[ii].size(), ' ');
    }
    H5Awrite(attr, atype, buf.c_str());
}

/** write an ArrayVectorView as an attribute with name to a location identifier
 */
template<class T>
inline void writeHDF5Attr(  hid_t loc,
                            const char* name,
                            ArrayVectorView<T>  & array)
{
    writeHDF5Attr(loc, name,
                  MultiArrayView<1, T>(MultiArrayShape<1>::type(array.size()),
                                       array.data()));
}

/** write an Attribute given a file and a path in the file.
 *  the path in the file should have the format
 *  [attribute] or /[subgroups/]dataset.attribute or
 *  /[subgroups/]group.attribute.
 *  The attribute is written to the root group, a dataset or a subgroup
 *  respectively
 */
template<class Arr>
inline void writeHDF5Attr(  std::string filePath,
                            std::string pathInFile,
                            Arr  & ar)
{
    std::string path_name(pathInFile), group_name, data_set_name, message, attr_name;
    std::string::size_type delimiter = path_name.rfind('/');

    //create or open file
    HDF5Handle file_id(createFile(filePath), &H5Fclose,
                       "writeToHDF5File(): unable to open output file.");

    // get the groupname and the filename
    if(delimiter == std::string::npos)
    {
        group_name    = "/";
        data_set_name = path_name;
    }

    else
    {
        group_name = std::string(path_name.begin(), path_name.begin()+delimiter);
        data_set_name = std::string(path_name.begin()+delimiter+1, path_name.end());
    }
    delimiter = data_set_name.rfind('.');
    if(delimiter == std::string::npos)
    {
        attr_name = path_name;
        data_set_name = "/";
    }
    else
    {
        attr_name = std::string(data_set_name.begin()+delimiter+1, data_set_name.end());
        data_set_name = std::string(data_set_name.begin(), data_set_name.begin()+delimiter);
    }
    
    HDF5Handle group(openGroup(file_id, group_name), &H5Gclose,
                     "writeToHDF5File(): Unable to create and open group. attr ver");

    if(data_set_name != "/")
    {
        HDF5Handle dset(H5Dopen(group, data_set_name.c_str(), H5P_DEFAULT), &H5Dclose,
                        "writeHDF5Attr():unable to open dataset");
        writeHDF5Attr(hid_t(dset), attr_name.c_str(), ar);
    }
    else
    {
        writeHDF5Attr(hid_t(group), attr_name.c_str(), ar);
    }

}
#endif
} // namespace vigra

#endif // VIGRA_HDF5IMPEX_HXX
