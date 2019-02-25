from skimage.external import tifffile
import os.path

def hack(input_tif):
    """
    This method allows to bypass the strange faulty behaviour of
    skimage.external.tifffile.imread() when it gets a list of paths or
    a glob pattern. This function extracts the image names and the path.
    Then, one can os.chdir(path) and call tifffile.imread(names),
    what will now behave well.
    """
    assert len(input_tif) > 0
    names = []
    path = str()
    for i in input_tif:
        names.append(os.path.basename(i))
    path = os.path.dirname(input_tif[0])
    return path, names