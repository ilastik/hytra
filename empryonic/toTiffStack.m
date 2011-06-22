function toTiffStack(h5filename, outdir, istransposed)
    raw = hdf5read(h5filename, '/raw/volume');
    outdir = [outdir '/'];
    if ~exist(outdir, 'dir'),
        mkdir(outdir);
    end;
    if ~istransposed,
        nStacks=length(raw(:,1,1));
    else
        nStacks=length(raw(1,1,:));
    end
    for ii = 150:179
%    for ii = 1:nStacks
        if ~istransposed,
            data = squeeze(raw(ii,:,:))';
        else
            data = raw(:,:,ii);
        end;
        imwrite(uint8(min(max(single(data)-100, 0)/4000*255, 255)), ...
            [outdir 'stack_' num2str(ii, '%05u') '.tiff'], 'tiff', ...
            'Compression','none')
    end
    