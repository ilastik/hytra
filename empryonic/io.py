import os.path as _path

import h5py
import numpy as np

import tracklets as _ts
import track as _track



def __loadDataset( filename, h5path):
    '''Load a dataset from a hdf5 file as a numpy array.

    filename: path to hdf5 file as a string
    h5path:   location of the data inside the hdf5 file
    '''
    f = h5py.File( filename, mode='r' )
    data = f[h5path].value
    f.close()
    return data
    
def __loadDatasets( filenames, h5path):
    '''Load the same dataset from different files.

    Works for a sequence of files or a single file. Returns just the
    dataset in the latter case.
    
    filenames: path to hdf5 file(s) as a string or a sequence of strings.
    h5path:   location of the data inside the hdf5 file
    '''
    if(isinstance(filenames, str)):
        return __loadDataset(filenames, h5path)
    else:
        def loader( filename ):
            return __loadDataset(filename, h5path)
        return map(loader, filenames)


def loadRaw( filenames, h5path = "raw/volume"):
    '''Load raw data from hdf5 file(s) as a numpy array.

    filenames: path to hdf5 file(s) as a string or a sequence of strings.
    h5path: location of the data inside the hdf5 file
    '''
    return __loadDatasets( filenames, h5path)

def iterRaw( filenames, h5path = "raw/volume" ):
    '''Similar to loadRaw, but returns an interator instead.'''
    for filename in filenames:
        yield __loadDatasets( filename, h5path )



def loadSegmentation( filenames, h5path = "segmentation/volume"):
    '''Load segmentation results from hdf5 file(s) as a numpy array.

    filenames: path to hdf5 file(s) as a string or a sequence of strings.
    h5path: location of the data inside the hdf5 file
    '''
    return __loadDatasets( filenames, h5path)

def iterSegmentation( filenames, h5path = "raw/volume" ):
    '''Similar to loadSegmentation, but returns an interator instead.'''
    for filename in filenames:
        yield __loadDatasets( filename, h5path )



def filenames(path, range = xrange(1,11)):
    '''Generate a list of filenames from a template.

    path: a filename template; for instance "/data/file_%03d.h5"
    range: list of numbers
    '''
    return map(lambda i: path % i, range)



def write_visbricks_file( h5_filenames, out_dir, visbricks_fn = "visbricks_time-dependent-h5.txt"):
    visbricks_fn = _path.join(out_dir, visbricks_fn)
    with open( visbricks_fn, 'w') as f:
        f.write("0 " + str(len(h5_filenames)-1) + "\n")
        for fn in h5_filenames:
            f.write(_path.abspath(fn)+"\n")



def tracklet_from_labelgroup( h5_labelgroup, timestep = None, position = 'mean', add_features_as_meta = True):
    '''Construct a Tracklet instance from a hdf5 group describing a featureset (a.k.a. "labelgroup").

    The id will be set to the name of the hdf5 labelgroup (converted to int; make sure it is convertible).

    h5_labelgroup - a h5py.Group instance
    timestep - Tracklet t coordinate 
    position - "mean" or "max" -> use either intensity weighted mean position or maximum intensity position of the connected
               component as the Tracklet coordinates.
    add_features_as_meta - If true, add all features as meta information to the Tracklet.
    '''
    if position not in ['mean', 'max']:
        raise ValueError('tracklet_from_labelgroup: invalid position: ' + str(position))
    pos_feat = 'com' if position == 'mean' else 'intmaxpos'

    if pos_feat not in h5_labelgroup.keys():
        raise Exception(position + " feature not present in label group " + str(h5_labelgroup))

    # read out coordinates
    if pos_feat == 'com':
        x,y,z = h5_labelgroup[pos_feat][0], h5_labelgroup[pos_feat][1], h5_labelgroup[pos_feat][2]
    else:
        x,y,z = h5_labelgroup[pos_feat][1], h5_labelgroup[pos_feat][2], h5_labelgroup[pos_feat][3]

    the_tracklet = _ts.Tracklet(x,y,z, timestep, int(_path.basename(h5_labelgroup.name)))

    # add features as meta
    if add_features_as_meta == True:
        def add_as_meta(name, obj):
            if isinstance(obj, h5py.Dataset):
                the_tracklet.meta[name] = obj.value        
        h5_labelgroup.visititems(add_as_meta)

    return the_tracklet



def ctracklet_from_labelgroup( h5_labelgroup ):
    the_tracklet = _track.cTracklet()
    
    # set tracklet id
    the_tracklet.ID = int(_path.basename(h5_labelgroup.name))

    # add features
    def add_feature(name, obj):
        if isinstance(obj, h5py.Dataset):
            the_tracklet.add_feature_array(name, len(obj.value))
            for i,v in enumerate(obj.value):
                the_tracklet.set_feature_value(name, i, float(v))
    h5_labelgroup.visititems(add_feature)

    return the_tracklet


    
class LineageH5( h5py.File ):
    mov_ds = "/tracking/Moves"
    mov_ener_ds = "/tracking/Moves-Energy"
    app_ds = "/tracking/Appearances"
    app_ener_ds = "/tracking/Appearances-Energy"
    dis_ds = "/tracking/Disappearances"
    dis_ener_ds = "/tracking/Disappearances-Energy"
    div_ds = "/tracking/Splits"
    div_ener_ds = "/tracking/Splits-Energy"
    feat_gn = "/features"
    track_gn = "/tracking/"

    # timestep will be set in loaded traxels accordingly
    def __init__( self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        if "timestep" in kwargs:
            self.timestep = kwargs["timestep"]
        else:
            self.timestep = 0

    def init_tracking( self, div=np.empty(0), mov=np.empty(0), dis=np.empty(0), app=np.empty(0)):
        if "tracking" in self.keys():
            del self["tracking"]
        self.create_group("tracking")

    def has_tracking( self ):
        if "tracking" in self.keys():
            return True
        else:
            return False
            
    def add_move( self, from_id, to_id):
        n_moves = self[self.mov_ds].shape[0];
        movs = self.get_moves()
        new = np.vstack([movs, (from_id, to_id)])
        self.update_moves(new)

    def update_moves( self, mov_pairs ):
        if _path.basename(self.mov_ds) in self[self.track_gn].keys():
            del self[self.mov_ds]
        if len(mov_pairs) > 0:
            self[self.track_gn].create_dataset("Moves", data=np.asarray( mov_pairs, dtype=np.int32))

    def get_moves( self ):
        if _path.basename(self.mov_ds) in self[self.track_gn].keys():
            return self[self.mov_ds].value
        else:
            return np.empty(0)
    def get_move_energies( self ):
        if _path.basename(self.mov_ener_ds) in self[self.track_gn].keys():
            e = self[self.mov_ener_ds].value
            if isinstance(e, np.ndarray):
                return e
            else:
                return np.array([e])
        else:
            return np.empty(0)
        

    def get_divisions( self ):
        if _path.basename(self.div_ds) in self[self.track_gn].keys():
            return self[self.div_ds].value
        else:
            return np.empty(0)

    def update_divisions( self, div_triples ):
        if _path.basename(self.div_ds) in self[self.track_gn].keys():
            del self[self.div_ds]
        if len(div_triples) > 0:
            self[self.track_gn].create_dataset("Splits", data=np.asarray( div_triples, dtype=np.int32))

    def get_division_energies( self ):
        if _path.basename(self.div_ener_ds) in self[self.track_gn].keys():
            e = self[self.div_ener_ds].value
            if isinstance(e, np.ndarray):
                return e
            else:
                return np.array([e])
        else:
            return np.empty(0)

    def get_disappearances( self ):
        if _path.basename(self.dis_ds) in self[self.track_gn].keys():
            dis = self[self.dis_ds].value
            if isinstance(dis, np.ndarray):
                return dis
            else:
                return np.array([dis])
        else:
            return np.empty(0)

    def update_disappearances( self, dis_singlets ):
        if _path.basename(self.dis_ds) in self[self.track_gn].keys():
            del self[self.dis_ds]
        if len(dis_singlets) > 0:
            self[self.track_gn].create_dataset("Disappearances", data=np.asarray( dis_singlets, dtype=np.int32))
        
    def get_disappearance_energies( self ):
        if _path.basename(self.dis_ener_ds) in self[self.track_gn].keys():
            e = self[self.dis_ener_ds].value
            if isinstance(e, np.ndarray):
                return e
            else:
                return np.array([e])
        else:
            return np.empty(0)


    def get_appearances( self ):
        if _path.basename(self.app_ds) in self[self.track_gn].keys():
            app = self[self.app_ds].value
            if isinstance(app, np.ndarray):
                return app
            else:
                return np.array([app])
        else:
            return np.empty(0)

    def update_appearances( self, app_singlets ):
        if _path.basename(self.app_ds) in self[self.track_gn].keys():
            del self[self.app_ds]
        if len(app_singlets) > 0:
            self[self.track_gn].create_dataset("Appearances", data=np.asarray( app_singlets, dtype=np.int32))

    def get_appearance_energies( self ):
        if _path.basename(self.app_ener_ds) in self[self.track_gn].keys():
            e = self[self.app_ener_ds].value
            if isinstance(e, np.ndarray):
                return e
            else:
                return np.array([e])
        else:
            return np.empty(0)

    def rm_appearance( self, id ):
        apps = self.get_appearances()
        if not id in apps:
            raise Exception("LineageH5::rm_appearance(): id %d not an appearance" % id)
        filtered = apps[apps!=id]
        b = np.empty(dtype=apps.dtype, shape=(filtered.shape[0], 1))
        b[:,0] = filtered[:]
        self.update_appearances( b )

    def rm_disappearance( self, id ):
        diss = self.get_disappearances()
        if not id in diss:
            raise Exception("LineageH5::rm_disappearance(): id %d not an disappearance" % id)
        filtered = diss[diss!=id]
        b = np.empty(dtype=diss.dtype, shape=(filtered.shape[0], 1))
        b[:,0] = filtered[:]
        self.update_disappearances( b )

    def get_ids( self ):
        features_group = self[self.feat_gn]
        labelcontent = features_group["labelcontent"].value
        valid_labels = (np.arange(len(labelcontent))+1)[labelcontent==1]
        return valid_labels
        
    def Tracklets( self , timestep=None, position='mean', add_features_as_meta=True):
        valid_labels = self.get_ids()
        features_group = self[self.feat_gn]
        tracklets = _ts.Tracklets([tracklet_from_labelgroup( features_group[str(label)], timestep=timestep, add_features_as_meta = add_features_as_meta, position=position ) for label in valid_labels])
        return tracklets

    def Traxels( self , timestep=None, position='mean', add_features_as_meta=True):
        return self.Tracklets( timestep, position, add_features_as_meta )

    def cTraxels( self, as_python_list=False ):
        # probe for objects group (higher io performance than features group)
        if 'objects' in self.keys():
            print "-> 'objects' format detected -> will use instead of 'features'"
            return self._cTraxels_from_objects_group( as_python_list )
        # use old 'features' format for traxels
        else:
            if as_python_list:
                raise Exception("LineageH5::cTraxels: old format -> can't return as python list (not implemented yet)")
            return self._cTraxels_from_features_group()

    def _cTraxels_from_objects_group( self , as_python_list = False):
        objects_g = self["objects"]
        features_g = self["objects/features"]
        ids = objects_g["meta/id"].value
        valid = objects_g["meta/valid"].value
        features = {}
        for name in features_g.keys():
            features[name] = features_g[name].value

        if as_python_list:
            ts = list()
        else:
            ts = _track.cTraxels()
        for idx, is_valid in enumerate(valid):
            if is_valid:
                tr = _track.cTraxel()
                tr.Id = int(ids[idx])
                tr.Timestep = self.timestep
                for name_value in features.items():
                    tr.add_feature_array(name_value[0], len(name_value[1][idx]))
                    for i,v in enumerate(name_value[1][idx]):
                        tr.set_feature_value(name_value[0], i, float(v))
                if as_python_list:
                    ts.append(tr)
                else:
                    ts.add_traxel(tr)
        return ts

    def _cTraxels_from_features_group( self ):
        features_group = self[self.feat_gn]
        labelcontent = features_group["labelcontent"].value
        invalid_labels = (np.arange(len(labelcontent))+1)[labelcontent==0]

        # note, that we used the ctracklet_from_labelgroup() here before, but had
        # to replace it by the following code due to bad performance

        ts = _track.cTraxels()
        # state machine for parsing features group
        class Harvester( object ):
            def __init__( self, invalid_labels=[], timestep=0):
                self.current_ctracklet = None
                self.timestep = timestep
                self.invalid_labels = map(int, invalid_labels )
                
            def __call__(self, name, obj):
                # name is the full path inside feature group
                # entering a new label group...
                if name.isdigit():
                    # store away the last cTraxel
                    if self.current_ctracklet != None:
                        ts.add_traxel(self.current_ctracklet)
                    if int(name) in self.invalid_labels:
                        self.current_ctracklet = None
                        print "invalid!"
                    else:
                        self.current_ctracklet = _track.cTraxel()
                        self.current_ctracklet.Id = int(name)
                        self.current_ctracklet.Timestep = self.timestep
                elif name == 'featurecontent' or name == 'labelcontent' or name == 'labelcount':
                    pass
                else:
                    feature_name = _path.basename(name)
                    self.current_ctracklet.add_feature_array(feature_name, len(obj.value))
                    for i,v in enumerate(obj.value):
                        self.current_ctracklet.set_feature_value(feature_name, i, float(v))
        harvest = Harvester(invalid_labels, self.timestep)
        features_group.visititems(harvest)

        return ts



        
import unittest as ut
import numpy as np
two_labels_fn = "test_data/io/two_labels.h5"

class Test_LineageH5( ut.TestCase ):
    def test_Tracklets( self ):
        with LineageH5( two_labels_fn, 'r' ) as f:
            trs = f.Tracklets()
            self.assertEqual( len(trs.the) , 2 )

    def test_cTraxels( self ):
        with LineageH5( two_labels_fn, 'r' ) as f:
            trs = f.cTraxels()
            self.assertEqual( len(trs) , 2 )


class Test_tracklet_from_labelgroup( ut.TestCase ):
    def setUp( self ):
        self.f = h5py.File( two_labels_fn )
        self.labelgroup_23 = self.f['/features/23']
        self.labelgroup_41 = self.f['/features/41']

    def test_default( self ):
        tr = tracklet_from_labelgroup( self.labelgroup_23 )
        self.assertAlmostEqual(tr.x, 480.346, places = 3)
        self.assertAlmostEqual(tr.y, 594.464, places = 3)
        self.assertAlmostEqual(tr.z, 80.469, places = 3)
        self.assertEqual( (tr.t, tr.id), (None, 23) )

        self.assertEqual( len(tr.meta), 16 )
        # some spot tests
        self.assertTrue( tr.meta.has_key("com") )
        self.assertTrue( tr.meta.has_key("intmaxpos") )
        self.assertTrue( np.all(tr.meta['intmaxpos'] == np.asarray([767.0, 480.0, 591.0, 79.0])))

    def test_max_position( self ):
        tr = tracklet_from_labelgroup( self.labelgroup_23, timestep=44, position='max' )
        self.assertEqual(tr.x, 480.0)
        self.assertEqual(tr.y, 591.0)
        self.assertEqual(tr.z, 79.0)
        self.assertEqual( (tr.t, tr.id), (44, 23) )
        self.assertEqual( len(tr.meta), 16 )

    def test_no_meta( self ):
        tr = tracklet_from_labelgroup( self.labelgroup_23, add_features_as_meta=False )
        self.assertEqual( len(tr.meta), 0 )

    def tearDown( self ):
        del self.labelgroup_23
        del self.labelgroup_41
        self.f.close()
        del self.f

        

if __name__=='__main__':
    ut.main()
