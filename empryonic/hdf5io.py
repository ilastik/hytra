import h5py
import numpy as np


def load_object_features(handle, features = ['volume','bbox','position','com','pc','intensity','intminmax','pair','sgf','lcom','lpc','lintensity','lintminmax','lpair','lsgf']):
    """
    Load the features given in 'features' of all objects in file 'handle' and return as
    a matrix of features.
    """
    
    featuregroup = '/objects/features'
    
    # create empty target array with correct number of objects
    featurevector = np.zeros([handle[featuregroup][features[0]][:].shape[0],0], dtype = np.float32)
    
    # read the features consecutively for all objects
    for f_name in features:
        f = handle[featuregroup][f_name][:]
        featurevector = np.append(featurevector,f,axis=1)
        
    featurevector[np.isnan(featurevector)] = 0
    return featurevector
    
    

def load_features_one_object(handle, index, features = ['volume','bbox','position','com','pc','intensity','intminmax','pair','sgf','lcom','lpc','lintensity','lintminmax','lpair','lsgf']):
    """
    Load the features given in 'features' of object 'index' in file 'handle' and return as
    a matrix of features.
    """
    allfeatures = load_object_features(handle, features)
    if index >= allfeatures.shape[0]:
        print "-> Error: given index %i could not be found in file."%index
        print "-> Aborted."
        sys.exit(1)
    
    fvector = np.zeros((1,allfeatures.shape[1]),dtype=np.float32)
    fvector[0,:] = allfeatures[index-1,:]

    return fvector

    

def load_single_feature(handle, featurename):
    """
    Load the feature 'featurename' of all objects in file 'handle' and return as
    a matrix of features.
    """
    
    featuregroup = '/objects/features'

    featurevector = handle[featuregroup][featurename][:]
    
    featurevector[np.isnan(featurevector)] = 0
    return featurevector



def set_one_label(handle, category, index, label):
    """
    General labeling function.
    Usecase: Give labels to a cell index, for example for labeling the 
    segmentation or the tracking.
    
    The labels are stored in /objects/meta/[category]. The dataset is
    is created if it doesn't exist.
    
    Input:
    handle: HDF5 file handle
    category: string naming the dataset for the labels
    index: index of the cell to be labeled (index > 0)
    label: int32, > 0
    """
    
    if '/objects/meta' in handle:
        group = handle['/objects/meta']
    else:
        print "-> Error: HDF5 file does not contain '/objects/meta' group."
        print "-> Aborted."
        sys.exit(1)

    if not category in group:
        lset = -np.ones((group['id'][:].shape[0],1), dtype = np.int32)
        lset[index-1] = label
        group[category] = lset
    else:
        lset = group[category]
        lset[index-1] = label



def get_one_label(handle, category, index):
    """
    Only get the label of the object with [index].
    """
    labels = get_labels(handle, category)
    return labels[index-1]



def get_labels(handle, category):
    """
    Get a label that has been assigned before.
    
    The labels are read from /objects/meta/[category].
    
    Input:
    handle: HDF5 file handle
    category: string naming the dataset for the labels
    index: index of the cell to be labeled (index > 0)

    Output: int32 label. returns -1 on error
    """
    if '/objects/meta' in handle:
        group = handle['/objects/meta']
    else:
        print "-> Error: HDF5 file does not contain '/objects/meta' group."
        print "-> Aborted."
        sys.exit(1)

    if not category in group:
        return -np.ones((group['id'][:].shape[0],1) ,dtype=np.int32)
    else:
        lset = group[category]
        return lset[:]



def set_labels(handle, category, labels):
    """
    Write a list of labels into the meta group in the HDF5 file.
    
    The labels are stored in /objects/meta/[category]. The dataset is
    is overwritten if it exists.
    
    Input:
    handle: HDF5 file handle
    category: string naming the dataset for the labels
    labels: array of labels
    """
    
    if '/objects/meta' in handle:
        group = handle['/objects/meta']
    else:
        print "-> Error: HDF5 file does not contain '/objects/meta' group."
        print "-> Aborted."
        sys.exit(1)

    if not category in group:
        lset = group.create_dataset(category, labels.shape, dtype = np.int32)
        lset[:] = labels
    else:
        lset = group[description]
        lset.resize(shape)
        lset[:] = labels.astype(np.int32)



def set_probabilities(handle, category, probabilities):
    """
    Write a list of probabilities into the meta group in the HDF5 file.
    
    The probabilities are stored in /objects/meta/[category]. The dataset is
    is overwritten if it exists.
    
    Input:
    handle: HDF5 file handle
    category: string naming the dataset for the probabilities
    probabilities: array of probabilities
    """
    
    if '/objects/meta' in handle:
        group = handle['/objects/meta']
    else:
        print "-> Error: HDF5 file does not contain '/objects/meta' group."
        print "-> Aborted."
        sys.exit(1)

    if not category in group:
        lset = group.create_dataset(category, probabilities.shape, dtype = np.float32)
        lset[:] = probabilities
    else:
        lset = group[description]
        lset.resize(shape)
        lset[:] = probabilities.astype(np.float32)


