from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
from skimage.external import tifffile

def hack(input_tif):
    """
    This method allows to bypass the strange faulty behaviour of
    skimage.external.tifffile.imread() when it gets a list of paths or
    a glob pattern. This function extracts the image names and the path.
    Then, one can os.chdir(path) and call tifffile.imread(name),
    what will now behave well.
    """
    name = []; path = str()
    for i in input_tif:
        name.append(i.split('/')[-1])
    path_split = list(input_tif)[0].split('/')[0:-1]
    for i in path_split:
        path += i+'/'
    return path, name