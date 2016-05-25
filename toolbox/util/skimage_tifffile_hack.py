from skimage.external import tifffile

def hack(input_tif):
    name = []; path = str()  # hack
    for i in input_tif:
        name.append(i.split('/')[-1])
    path_split = list(input_tif)[0].split('/')[0:-1]
    for i in path_split:
        path += i+'/'
    return path, name
    # label_volume = tifffile.imread(name)