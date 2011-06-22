% function to create artificial cell data and a corresponding segmentation
%
% input
%
% prefix            path with filename prefix to store the data
% size              size of the volume (cubic)
% nuclei            number of nuclei to be created in the volume
%
% (C)2010 Martin Lindner
%
function create_artificial_data(prefix, size, nuclei)
% Create some artificial data

%Attention! For n > 600 the Output file will be more than 2GB in size, for 
%processing in MATLAB you need at least 8GB of RAM!!!
for size = size
    for nuclei_number = nuclei
        n = size; %create n*n*n cube
        
        % create volume filled with zeros
        aVol = zeros(n,n,n);
        
        % fill p nuclei to the volume
        p = nuclei_number;
        for i = 1:p
            %select random pixel and assign random intensity (range: 0..10000)
            aVol(randi(n),randi(n),randi(n)) = rand*10000;
        end
        
        % blurr points to look like cells
        gauss = exp(-(-4:4).^2/2);
        aVol = convn(aVol,reshape(gauss,9,1,1),'same');
        aVol = convn(aVol,reshape(gauss,1,9,1),'same');
        aVol = convn(aVol,reshape(gauss,1,1,9),'same');
        
        % threshold volume to get some 'segmentation'
        aNuclei = uint16(bwlabeln(aVol > 10));
        filename = [prefix,int2str(n),'/simdata_',int2str(n),'_',int2str(p),'_raw.h5'];
        
        % write with non-builtin routine. Vigra does not like chunked data.
        writeContiguousHDF5(filename,'/raw/volume',uint16(aVol),0);
        writeContiguousHDF5(filename','/segmentation/volume',aNuclei,0);
        
    end
end


