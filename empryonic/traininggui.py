from __future__ import absolute_import
from __future__ import unicode_literals
from mayavi import mlab
import h5py
import numpy as np
import vigra
import time

from . import trainingcore
from . import visCell
from . import hdf5io

from traits.api import *
from traitsui.api import *

from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from tvtk.pyface.scene_editor import SceneEditor 

class msgdialog(HasTraits):
    message = Label("")
    
    def __init__(self, message):
        self.message = message
    
    view = View( Item('message',show_label=False,style='readonly'),
                 buttons = ['OK'], 
                 title='Message')



def showmessage(message):
    msg = msgdialog(message)
    msg.configure_traits()
    


class traininggui(HasTraits):
  
    """ Group: File selection """
    fileIndex = Int(0)
    btnLoadFile = Button()
    btnLoadNextFile = Button()
    btnLoadPreviousFile = Button()
    btnNextHere = Button()
    btnPrevHere = Button()
    
    """ Info """
    numFiles = Int(0)
    numObjects = Int(0)
    numLabels = Int(0)
    numClasses = Int(0)
    classCount = Array(dtype=np.int32)

    """ Group: Jump navigation """
    btnJump = Button()
    btnJumpRandom = Button()
    index = Int(1)

    """ Group: Forward navigation """
    btnShowNext = Button()
    btnShowPrevious = Button()
    btnShowNextLabel = Button()
    btnShowPreviousLabel = Button()
    nextLabel = Int(0)
    btnShowNextUncertain = Button()

    """ Group: Labeling"""
    label = Int(-1)
    btnZero = Button()
    btnOne = Button()
    btnLblAs = Button()
    labelAs = Int(1)

    """ Group: Settings """
    borderSize = Int(30)
    datasetTraining = Str('training')
    datasetPrediction = Str('prediction')
    useRandomForest = Bool(False)
    randomForestTrees = Int(255)
    
    """ Group: Train&Predict """
    btnTrainIntern = Button()
    extLabelSource = Str('training')
    btnTrainExtern = Button()
    
    predDestination = Str('prediction')
    predType = Enum('Class', 'Class Probabilities')
    btnPredict = Button()
    
    btnSaveRF = Button()
    btnLoadRF = Button()
    RFFile = Str('RandomForest.h5')
    
    """ Scene views """    
    scene3d = Instance(MlabSceneModel, ())
    scene2d = Instance(MlabSceneModel, ())



    def __init__(self, files):
        self.files = files
        self.fileIndex = 0
        self.t = trainingcore.trainingcore(self.files[self.fileIndex], scene3d=self.scene3d, scene2d=self.scene2d)
        self.t.jump_to_cell(self.index)
        self.prepareRF()
        self.update_info()
        self.seriousRF = None

            

    def _btnLoadFile_fired(self):
        if self.fileIndex >= 0 and self.fileIndex < len(self.files):
            self.t = trainingcore.trainingcore(self.files[self.fileIndex], self.index)
            self.t.jump_to_cell(self.index)
            self.update_info()
            self.t.borderSize = self.borderSize
        else:
            showmessage("Error: Cannot view this time step - index out of bounds.")



    def _btnLoadNextFile_fired(self):
        if self.fileIndex+1 < len(self.files):
            self.fileIndex = self.fileIndex +1
            self.t = trainingcore.trainingcore(self.files[self.fileIndex], self.index)
            self.t.jump_to_cell(self.index)
            self.update_info()
            self.t.borderSize = self.borderSize
        else:
            showmessage("Error: Cannot load next time step - index out of bounds.")


    def _btnLoadPreviousFile_fired(self):
        if self.fileIndex > 0:
            self.fileIndex = self.fileIndex -1
            self.t = trainingcore.trainingcore(self.files[self.fileIndex], self.index)
            self.t.jump_to_cell(self.index)
            self.update_info()
            self.t.borderSize = self.borderSize
        else:
            showmessage("Error: Cannot load previous time step - index out of bounds.")



    def _btnNextHere_fired(self):
        if self.fileIndex+1 < len(self.files):
            currentBBox = self.t.currentBBox
            glanceFile = self.files[self.fileIndex+1]
            t = trainingcore.trainingcore(glanceFile)
            t.borderSize = self.borderSize
            t.show_volume(currentBBox)
            t.currentBBox = currentBBox
        else:
            showmessage("Error: Cannot view next time step - already reached last file.")


    def _btnPrevHere_fired(self):
        if self.fileIndex-1 >= 0:
            currentBBox = self.t.currentBBox
            glanceFile = self.files[self.fileIndex-1]
            t = trainingcore.trainingcore(glanceFile)
            t.borderSize = self.borderSize
            t.show_volume(currentBBox)
            t.currentBBox = currentBBox
        else:
            showmessage("Error: Cannot view previous time step - already reached first file.")
        

    def _btnShowNext_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.show_next_cell()
        self.label = self.find_label(self.index)



    def _btnShowPrevious_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.show_previous_cell()
        self.label = self.find_label(self.index)



    def _btnShowNextLabel_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.show_next_label_cell(self.datasetPrediction,self.nextLabel)
        self.label = self.find_label(self.index)



    def _btnShowPreviousLabel_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.show_previous_label_cell(self.datasetPrediction,self.nextLabel)
        self.label = self.find_label(self.index)



    def _btnShowNextUncertain_fired(self):
        if self.useRandomForest and self.RF != None:
            self.t.RF = self.RF
            self.index = self.t.show_next_uncertain_cell()
            self.label = self.find_label(self.index)
        else:
            self.t.RF = None
            showmessage("Error: Cannot find cells with high uncertainty. Need a pretrained Random Forest.")
       


    def _btnOne_fired(self):
        with h5py.File(self.files[self.fileIndex], 'a') as f:
            hdf5io.set_one_label(f,self.datasetTraining,self.index,1)
            newf = hdf5io.load_features_one_object(f,self.index)
        
        self.labels = np.append(self.labels,[[1]],axis=0).astype(np.int32)

        self.features = np.append(self.features, newf, axis=0).astype(np.float32)
        
        if self.useRandomForest and self.RF != None:
            self.RF.learnRF(self.features, self.labels.astype(np.uint32))
            self.t.RF = self.RF
        else:
            self.t.RF = None
        
        self.index = self.t.show_next_cell()
        self.label = self.find_label(self.index)
        self.update_info()



    def _btnZero_fired(self):
        with h5py.File(self.files[self.fileIndex], 'a') as f:
            hdf5io.set_one_label(f,self.datasetTraining,self.index,0)
            newf = hdf5io.load_features_one_object(f,self.index)
        
        self.labels = np.append(self.labels,[[0]],axis=0).astype(np.int32)

        self.features = np.append(self.features, newf, axis=0).astype(np.float32)
        
        if self.useRandomForest and self.RF != None:
            self.RF.learnRF(self.features, self.labels.astype(np.uint32))
            self.t.RF = self.RF
        else:
            self.t.RF = None
        
        self.index = self.t.show_next_cell()
        self.label = self.find_label(self.index)
        self.update_info()



    def _btnLblAs_fired(self):
        with h5py.File(self.files[self.fileIndex], 'a') as f:
            hdf5io.set_one_label(f,self.datasetTraining,self.index,self.labelAs)
        
        if self.labelAs >= 0:
            self.labels = np.append(self.labels,[[self.labelAs]],axis=0).astype(np.int32)
            with h5py.File(self.files[self.fileIndex], 'r') as f:
                newf = hdf5io.load_features_one_object(f,self.index)
            self.features = np.append(self.features, newf, axis=0).astype(np.float32)
        
        if self.useRandomForest and self.RF != None:
            self.RF.learnRF(self.features, self.labels.astype(np.uint32))
            self.t.RF = self.RF
        else:
            self.t.RF = None
        
        self.index = self.t.show_next_cell()
        self.label = self.find_label(self.index)
        self.update_info()



    def _btnJump_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.jump_to_cell(self.index)
        self.label = self.find_label(self.index)



    def _btnJumpRandom_fired(self):
        if self.useRandomForest:
            self.t.RF = self.RF
        else:
            self.t.RF = None
        self.index = self.t.jump_to_cell(np.random.randint(1,self.t.maxLabel+1))
        self.label = self.find_label(self.index)

    

    def _btnTrainIntern_fired(self):
        """
        """
        if np.unique(self.labels).shape[0] >= 2:
            self.seriousRF = vigra.learning.RandomForest(treeCount=self.randomForestTrees)
            self.seriousRF.learnRF(self.features, self.labels.astype(np.uint32))
            showmessage("Training successful. Used %i trees and %i training samples of %i classes."%(self.randomForestTrees,self.labels.shape[0],np.unique(self.labels).shape[0]))
        else:
            showmessage("Error: Not enough labels provided. Provide labels of at least two classes!")



    def _btnTrainExtern_fired(self):
        """
        """
        with h5py.File(self.files[self.fileIndex], 'r') as f:
            l = hdf5io.get_labels(f,self.extLabelSource).astype(np.int32)
        indices = l >= 0
        l = l[indices]

        if np.unique(l).shape[0] >= 2:
            with h5py.File(self.files[self.fileIndex], 'r') as fl:
                f = hdf5io.load_object_features(fl)[indices,:]
            self.seriousRF = vigra.learning.RandomForest(treeCount=self.randomForestTrees)
            self.seriousRF.learnRF(f, l.astype(np.uint32))
            showmessage("Training successful. Used %i trees and %i training samples of %i classes."%(self.randomForestTrees,l.shape[0],np.unique(l).shape[0]))
        else:
            showmessage("Error: Not enough labels found in dataset. Provide labels of at least two classes!")



    def _btnSaveRF_fired(self):
        """
        """
        if self.seriousRF != None and self.RFFile != "":
            self.seriousRF.writeHDF5(str(self.RFFile))
            showmessage('Successfully wrote Random Forest to file %s'%self.RFFile)
        else:
            showmessage('Error: train a Random Forest first and provide a filename.')



    def _btnLoadRF_fired(self):
        """
        """
        if self.RFFile != "":
            self.seriousRF = vigra.learning.RandomForest(str(self.RFFile))
            showmessage('Successfully loaded Random Forest from file %s'%self.RFFile)
        else:
            showmessage('Error: provide a filename.')



    def _btnPredict_fired(self):
        """
        """
        if self.seriousRF != None:
            with h5py.File(self.files[self.fileIndex], 'a') as fl:
                f = hdf5io.load_object_features(fl)
            if self.predType == 'Class':
                l = self.seriousRF.predictLabels(f)
                with h5py.File(self.files[self.fileIndex], 'a') as fl:
                    hdf5io.set_labels(fl,self.predDestination,l)
                showmessage('Successfully predicted labels of %i objects.'%f.shape[0])
            if self.predType == 'Class Probabilities':
                l = self.seriousRF.predictProbabilities(f)
                with h5py.File(self.files[self.fileIndex], 'a') as fl:
                    hdf5io.set_probabilities(fl,self.predDestination,l)
                showmessage('Successfully predicted probabilities of %i objects.'%f.shape[0])
        else:
            showmessage('Error: train the Random Forest first.')



    def find_label(self, index):
        """
        Returns the predicted label. If not available, it returns the
        trained label. If no label is found, -1 is returned.
        """
        with h5py.File(self.files[self.fileIndex], 'r') as f:
            l = hdf5io.get_labels(f,self.datasetPrediction)[index-1,0]
            if l == -1:
                l = hdf5io.get_labels(f,self.datasetTraining)[index-1,0]
        return l
    
    
    
    def update_info(self):
        """
        Update the values in the info field.
        """
        uniquelabels = np.unique(self.labels)
        count = np.zeros((uniquelabels.shape[0],2),dtype=np.int32)
        count[:,0] = uniquelabels[:]
        for i in range(uniquelabels.shape[0]):
            count[i,1] = np.sum(self.labels == uniquelabels[i])
        
        self.numFiles = len(self.files)
        self.numObjects = self.t.maxLabel
        self.numLabels = self.labels.shape[0]
        self.numClasses = len(uniquelabels)
        self.classCount = count
    
    
    
    def _borderSize_changed(self):
        """
        Apply a changed border size immediately
        """
        self.t.borderSize = self.borderSize


    def prepareRF(self):
        # prepare the random forest based on the features in h5file
        self.RF = vigra.learning.RandomForest(treeCount=80)
        with h5py.File(self.files[self.fileIndex], 'r') as f:
            nfeatures = hdf5io.load_features_one_object(f, 1).size
        self.features = np.zeros([0,nfeatures], dtype=np.float32)
        self.labels = np.zeros([0,1], dtype=np.int32)


    view = View(VSplit(
                    HSplit(
                        Item(name='scene3d', 
                            editor=SceneEditor(scene_class=MayaviScene),
                            show_label=False,
                            height = 500
                        ),
                        Item(name='scene2d', 
                            editor=SceneEditor(scene_class=MayaviScene),
                            show_label=False,
                            height = 500
                        ),
                    ),
                    HGroup(
                        Tabbed(
                            Group(
                                Item('fileIndex', label='Timestep' ),
                                '_',
                                HGroup(
                                    Item('btnLoadPreviousFile', label='<<<', show_label=False, tooltip="Load the previous time step." ),
                                    Item('btnLoadFile', label='Load This File', show_label=False, tooltip="Load/switch back to the selected time step." ),
                                    Item('btnLoadNextFile', label='>>>', show_label=False, tooltip="Loat the next time step." ), 
                                    
                                ),
                                HGroup(
                                    Item('btnPrevHere', label='Glance previous', show_label=False, tooltip="Show the same position in the previous time step." ),
                                    Item('btnNextHere', label='Glance next', show_label=False, tooltip="Show the same position in the next time step." ), 
                                    
                                ),
                                '_',
                                Label('- Info -'),
                                Item('numFiles', label='Number of Files', style='readonly'),
                                Item('numObjects', label='Number of Objects', style='readonly'),
                                Item('numLabels', label='Number of Labels', style='readonly'),
                                Item('numClasses', label='Number of Classes', style='readonly'),
                                Item('classCount', label='Labels per Class', style='readonly'),
                                
                                label = 'File'
                            ),
                            Group(
                                Item('borderSize', label='Neighborhood Size', tooltip="Size of the neighborhood displayed around the object." ),
                                Item('datasetTraining', label='Write Labels' ),
                                Item('datasetPrediction', label='Read Labels' ),
                                Item('useRandomForest', label='Interactive Labeling', tooltip="Enable interactive labeling using a Random Forest classifier. This requires that labels for at least two different classes are provided."  ),
                                Item('randomForestTrees', label='RF Trees', tooltip="Number of trees used for the serious Random Forests." ),
                                label = "Settings"
                            ),
                        ),
                        Tabbed(
                            HGroup(
                                Group(
                                    HGroup(
                                        Item('btnShowPrevious', label='<<<', show_label=False, tooltip="Decrease object index by one."),
                                        Item('index', label='Selected index', show_label=False, tooltip="Object index."),
                                        Item('btnShowNext', label='>>>', show_label=False, tooltip="Increase object index by one."),
                                    ),                                    
                                    HGroup(
                                        Item('btnJump', label='Jump to index', show_label=False, tooltip="Jump to object index given in the field above." ), 
                                        Item('btnJumpRandom', label='Random index', show_label=False, tooltip="Jump to some ranodm object index." ), 
                                    ),
                                    '_',
                                    Label('Find object with label:'),
                                    HGroup(
                                        Item('btnShowPreviousLabel',label='<<<',show_label=False, tooltip="Search for label in backward direction."),
                                        Item('nextLabel', label='Next label',show_label=False, tooltip="A label name to search for, e.g. '0'. Searches in the dataset specified in 'Read Labels'."),
                                        Item('btnShowNextLabel',label='>>>',show_label=False, tooltip="Search for label in forward direction."),
                                    ),
                                    '_',
                                    Label('Suggest:'),
                                    Item('btnShowNextUncertain',label='Suggest object',show_label=False, tooltip="Let the Random Forest decide which object to present next."),
                                ),
                                '_',
                                Group(
                                    HGroup(
                                        Label('Current label: '),
                                        Item('label', style='readonly', show_label=False, tooltip="Label assigned to selected object."),
                                    ),
                                    HGroup(
                                        Item('btnZero', label='Label as 0', show_label=False, tooltip="Assign label 0 to object." ),
                                        Item('btnOne', label='Label as 1', show_label=False, tooltip="Assign label 1 to object." ),
                                    ),
                                    HGroup(
                                        Item('btnLblAs', label='Label as ...', show_label=False, tooltip = "Assign a custom label to the object." ),
                                        Item('labelAs',label='Label',show_label=False, tooltip="Provide a custom label >= 0."),
                                    ),
                                ),
                                label = "Controls"
                            ),
                            Group(
                                Item('btnTrainIntern',label='Train with recorded labels',show_label=False, tooltip="Use the labels recorded in this session.", width=150),
                                HGroup(
                                    Item('btnTrainExtern',label='Train with label dataset',show_label=False, tooltip="Use labels loaded from the dataset.", width=150),
                                    Item('extLabelSource', label='Label dataset', tooltip = "Dataset with labels. Stored in '/objects/meta'." ),

                                ),
                                '_',
                                HGroup(
                                    Item('predDestination', label='Write prediction to', tooltip = "Write the prediction in this dataset in '/objects/meta'." ),
                                    Item('predType', label='Prediction type', tooltip = "'Class' predicts only the class for each object, 'Probabilities' predicts the probabilities for each class." ),
                                ),
                                Item('btnPredict',label='Predict',show_label=False, tooltip="Calculate predictions for the opened file."),
                                '_',
                                Item('RFFile', label='Random Forest filename', tooltip = "HDF5 filename of Random Forest file. Random Forest must be stored in '/'." ),
                                HGroup(
                                    Item('btnSaveRF', label='Save Random Forest to file',show_label=False, tooltip = "Write Random Forest to HDF5 file." ),
                                    Item('btnLoadRF', label='Load Random Forest from file',show_label=False, tooltip = "Load Random Forest from HDF5 file." ),
                                ),
                                label = "Train&Predict"
                            ),
                        ),
                     ),
                ),
                title = 'Training GUI',
                resizable=True, 
                height=0.9, 
                width=0.9,
                )
