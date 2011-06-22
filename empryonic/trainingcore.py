from enthought.mayavi import mlab
import h5py
import numpy as np
import vigra
import time

import visCell
import hdf5io


from enthought.traits.api import *
from enthought.traits.ui.api import *



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
    


class trainingcore():
    borderSize = 30
    RF = None
    color = (0.5,1,0)
       
    def __init__(self, filehandle, index = 1, scene3d=None, scene2d=None):
        """
        initialize variables, load data, etc.
        """
        self.h5file = filehandle

        self.maxLabel = self.h5file["features"]["labelcount"][0]
        self.isValid = self.h5file["features"]["labelcontent"][:]
        self.shape = self.h5file["segmentation"]["labels"].shape
        self.bbox = hdf5io.load_single_feature(self.h5file, "bbox")
        self.index = index
        
        self.features = hdf5io.load_object_features(self.h5file)

        self.currentBBox = np.zeros((2,3))
        
        self.set_std_color()
        
        self.scene3d = scene3d
        self.scene2d = scene2d



    def is_valid(self, index):
        """
        Determine whether a given index corresponds to a valid cell.
        """
        return (index > 0) and (index <= self.maxLabel) and (self.isValid[index-1] == 1)


   
    def get_neighbors(self, seg, reference):
        """
        returns the indices of all cell labels in 'seg' except 'reference'
        """
        labels = np.unique(seg)
        for i in range(0,labels.size):
            if labels[i] == reference[0]:
                labels[i] = 0
        labels = np.unique(labels)
        if(labels[0] == 0):
            labels = labels[1:]
        
        return np.int16(labels)
        
    

    def show_cell(self, index):
        """
        show cell with label 'index', its surrounding, raw data, etc.
        """
        # create an array containing 'index'
        index = np.array([index], dtype=np.int32)
        
        # set the bounding box of the volume to be displayed
        bbox = self.bbox[index-1,:].squeeze()

        self.currentBBox = bbox

        #enlarge the bounding box
        # make sure that bbox[0..2] >= 0 (no array bound violation)

        bbox[0] = bbox[0] - min(self.borderSize, bbox[0]) 
        bbox[1] = bbox[1] - min(self.borderSize, bbox[1])
        bbox[2] = bbox[2] - min(self.borderSize, bbox[2])
        # make sure that bbox[3..5] <= self.shape[0] (no array bound violation)
        bbox[3] = bbox[3] + min(self.borderSize, self.shape[2]-bbox[3]) 
        bbox[4] = bbox[4] + min(self.borderSize, self.shape[1]-bbox[4])
        bbox[5] = bbox[5] + min(self.borderSize, self.shape[0]-bbox[5])
        
        # load the data
        seg1 = self.h5file["segmentation"]["labels"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        raw = self.h5file["raw"]["volume"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        
        print "Drawing cell number",index
        t0 = time.time()
        
        #draw everything
        fig1 = mlab.figure(1, size=(500,450))
        mlab.clf(fig1)
        visCell.drawImagePlane(fig1, raw, 'gist_ncar')
        visCell.drawVolumeWithoutReferenceCell(fig1, seg1, index, (0,0,1),0.2)
        visCell.drawReferenceCell(fig1, seg1, index, self.color, 0.4)
        
        fig2 = mlab.figure(2, size=(500,450))
        mlab.clf(fig2)
        visCell.draw2DView(fig2, raw[20:-20,:,:], seg1[20:-20,:,:], index, self.color)
        
        t = time.time() - t0
        print "Time for drawing:",t



    def show_volume(self, bbox):
        """
        show the volume with the given bounding box
        """
        # load the data
        seg1 = self.h5file["segmentation"]["labels"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        raw = self.h5file["raw"]["volume"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        
        print "Drawing volume"
        t0 = time.time()
        
        #draw everything
        fig1 = mlab.figure(1, size=(500,450))
        mlab.clf(fig1)
        visCell.drawImagePlane(fig1, raw, 'gist_ncar')
        visCell.drawVolumeWithoutReferenceCell(fig1, seg1, np.array((-1,)), (0,0,1),0.5)
        visCell.drawLabels(fig1, self.h5file, seg1, bbox)

        fig2 = mlab.figure(2, size=(500,450))
        mlab.clf(fig2)

        visCell.draw2DView(fig2, raw[20:-20,:,:], seg1[20:-20,:,:], -1)
        
        t = time.time() - t0
        print "Time for drawing:",t


    

    def show_next_cell(self):
        """
        Finds the next valid cell and displays it
        """
        nextIdx = 0
        for idx in range(self.index+1, self.maxLabel+1):
            if self.is_valid(idx):
                nextIdx = idx
                break
        if nextIdx > 0:
            self.index = nextIdx 
            self.set_cellness_color()
            self.show_cell(self.index)
            return nextIdx
        else:
            showmessage("No valid cells with higher indices available") 
            return self.index


    def show_previous_cell(self):
        """
        Finds and displays the previous valid cell
        """
        prevIdx = 0
        for idx in range(self.index-1, 0, -1):
            if self.is_valid(idx):
                prevIdx = idx
                break
        
        if prevIdx > 0:
            self.index = prevIdx
            self.set_cellness_color()
            self.show_cell(self.index)
            return prevIdx
        else:
            showmessage("No valid cells with lower indices available") 
            return self.index



    def show_next_label_cell(self, category, label):
        """
        Finds and displays the next valid cell labeled as 'label'
        """
        labels = hdf5io.get_labels(self.h5file, category)

        nextIdx = 0
        for idx in range(self.index+1, self.maxLabel+1):
            if labels[idx-1] == label:
                nextIdx = idx
                break

        if nextIdx > 0:
            self.index = nextIdx 
            self.set_cellness_color()
            self.show_cell(self.index)
            return nextIdx
        else:
            showmessage("No valid cells with higher indices available") 
            return self.index



    def show_previous_label_cell(self, category, label):
        """
        Finds and displays the previous valid cell labeled as 'label'
        """
        labels = hdf5io.get_labels(self.h5file, category)

        nextIdx = 0
        for idx in range(self.index-1, 0, -1):
            if labels[idx-1] == label:
                nextIdx = idx
                break

        if nextIdx > 0:
            self.index = nextIdx 
            self.set_cellness_color()
            self.show_cell(self.index)
            return nextIdx
        else:
            showmessage("No valid cells with lower indices available") 
            return self.index
             
        

    def show_next_uncertain_cell(self):
        """
        Finds and displays the next valid cell with high uncertainty
        """
        nextIdx = 0
        
        for idx in range(self.index+1, self.maxLabel+1):
            cn = self.get_cellness(idx)
            if cn > 0.2 and cn < 0.8:
                nextIdx = idx
                break
        
        if nextIdx > 0:
            self.index = nextIdx 
            self.set_cellness_color()
            self.show_cell(self.index)
            return nextIdx
        else:
            showmessage("No valid cells with higher indices available") 
            return self.index


    def jump_to_cell(self, index):
        """
        Jumps to provided index, if index is valid. Returns the new index if
        jump was successful and the old index otherwise.
        """
        if self.is_valid(index):
            self.index = index
            self.set_cellness_color()
            self.show_cell(self.index)
            return index
        else:
            showmessage("User-defined index does not correspond to valid cell") 
            return self.index
            


    def set_cellness_color(self):
        """
        Determine the color of the selected cell, if a Random Forest allows the prediction of the cellness.
        Use the standard color otherwise
        """
        if self.RF != None:
            cn = self.get_cellness()
            print "cellness: ",cn
            self.color = (float((1-abs(1-2*cn))**2+np.max([0,2*(0.5-cn)])),float(2*abs(cn-0.5)),float(0))
        else:
            self.set_std_color()        



    def set_std_color(self):
        """
        Set the color of the selected cell to a standard color.
        """
        self.color = (0.5,1,0)



    def get_cellness(self, index = -1):
        """
        Return the cellness of an object.
        """
        if index == -1:
            index = self.index
        feats = hdf5io.load_features_one_object(self.h5file, index)
        probs = self.RF.predictProbabilities(feats)
        return probs[0][1]


