% function that stores double matrices in an HDF5 file where a contiguous
% layout is used
%
% input
%
% filepath          string to the hdf5 file, e.g. c:\temp\myfile.h5
% datasetname       name of dataset inside HDF5 file, e.g. data
% data              a matrix in arbitrary dimensions
% rowMajor          0 or 1 corresponding to row or column major order
%
% (C)2009 Michael Hanselmann
%
function writeContiguousHDF5(filepath, datasetname, data, rowMajor)

% dimensions
dims = size(data);

% permute data?
if(rowMajor)
    data = permute(data, length(dims):-1:1);
end

if(exist(filepath))
    file = H5F.open(filepath, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
else
    file = H5F.create(filepath, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
end

dataset = datasetname;

%check dataset path for delimiter characters
if(dataset(1)~='/')
    dataset = ['/',dataset];  %always start in root directory
end
delimiter = strfind(dataset,'/');


%separate group and dataset name
if(max(size(delimiter))==1)&&(delimiter(1) == 1)
    group_name = '/'; %set group to root
    data_name = dataset;
else
    group_name = dataset(1:delimiter(max(size(delimiter))));
    data_name = dataset((delimiter(max(size(delimiter)))+1):end);
end

%create groups and sub-groups
ID = file;
for i = 1:(max(size(delimiter))-1)
    gname = group_name((delimiter(i)+1):(delimiter(i+1)-1));
    try
        ID = H5G.create(ID,gname,-1); %try to create group
    catch me_if_you_can
        ID = H5G.open(ID,gname); %open group, if it already exists
    end
end

dcpl = H5P.create('H5P_DATASET_CREATE');
H5P.set_layout(dcpl, 'H5D_CONTIGUOUS');
space = H5S.create_simple(length(dims), dims, dims);

data_type = 'H5T_IEEE_F64LE';
if(isa(data,'int32'))
    data_type = 'H5T_NATIVE_INT';
end
if(isa(data,'uint32'))
    data_type = 'H5T_INTEL_U32';
end
if(isa(data,'int16'))
    data_type = 'H5T_NATIVE_SHORT';
end
if(isa(data,'uint16'))
    data_type = 'H5T_INTEL_U16';
end
if(isa(data,'int64'))
    data_type = 'H5T_NATIVE_LONG';
end
if(isa(data,'double'))
    data_type = 'H5T_NATIVE_DOUBLE';
end
if(isa(data,'single'))
    data_type = 'H5T_NATIVE_FLOAT';
end

try
    dset = H5D.create(ID, data_name, data_type, space, dcpl); %create dataset
catch me_if_you_can
    dset = H5D.open(ID,data_name); %overwrite if dataset already exists
end

H5D.write (dset, data_type, 'H5S_ALL', space, 'H5P_DEFAULT', data);
space.close();
dcpl.close();
dset.close();
file.close();

end