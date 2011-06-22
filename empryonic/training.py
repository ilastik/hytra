#!/usr/bin/env python

from enthought.mayavi import mlab
import h5py
import numpy as np
import time

from enthought.traits.api import *
from enthought.traits.ui.api import View, Item, ButtonEditor

"""
README:

to run the training or the visualization only, just type:

import training
training.run(file1,file2)

It is assumed, that file1 and file2 are adjacent time frames and the tracking is present in the later
one. The labels are saved as plain text.

"""

class training():
    """
    This class allows to view the segmentation and tracking of two adjacent time frames.
    
    Usage:
    ------
    ###Initialization:

    from training import training
    t = training(filename1, filename2)
    
    ###Show next possible cell:
    
    t.showNextCell()
    
    ###Show cell number 'x'
    t.showCell(x)
    """
    def __init__(self, timestep1, timestep2, bsize = 60):
        """
        initialize variables, load data, etc.
        """
        self.file1 = h5py.File(timestep1, 'r')
        self.file2 = h5py.File(timestep2, 'r')
    	if "Moves" in self.file2["tracking"]:
            self.moves = self.file2["tracking"]["Moves"][:]
        else:
            self.moves = np.zeros( (0, 2), dtype='int32')
        if "Splits" in self.file2["tracking"]:
            self.splits = self.file2["tracking"]["Splits"][:]
        else:
            self.splits = np.zeros( (0, 3), dtype='int32')
        if "Disappearances" in self.file2["tracking"]:
            self.death = self.file2["tracking"]["Disappearances"][:]
        else:
            self.death = np.zeros( 0, dtype='int32')
        self.file1MaxLabel = self.file1["features"]["supervoxels"][0]
        self.file2MaxLabel = self.file2["features"]["supervoxels"][0]
        self.file1LabelContent = self.file1["features"]["labelcontent"][:]
        self.file2LabelContent = self.file2["features"]["labelcontent"][:]
        self.shape = self.file1["segmentation"]["labels"].shape
        if self.file2["segmentation"]["labels"].shape != self.shape:
            raise RuntimeError("Mismatching label volume sizes") 
        self.index = 0
        self.borderSize = bsize

    """
    def __del__(self):
        self.file1.close()
        self.file2.close()
    """    

    def isValid(self, index, forFirstSlice=True):
        """Determine whether a given index corresponds to a valid cell.
        Usage: t.isValid(index, forFirstSlice)
        If forFirstSlice == True [default], this routine checks whether the 
        given index corresponds to a valid cell for the first slice; otherwise
        for the second of the two slices.
        """
        if forFirstSlice:
            return (index > 0) and (index <= self.file1MaxLabel) and (self.file1LabelContent[index-1] == 1)
        else:
            return (index > 0) and (index <= self.file2MaxLabel) and (self.file2LabelContent[index-1] == 1)

    def findNextIndex(self):
        """
        Determine a valid cell to be displayed next. If no valid cells with higher indices
        than the current one exist, 0 is returned (note that cell indices start from 1 and
        not from 0)
        """
        nextIdx = 0
        for idx in range(self.index+1, self.file1MaxLabel+1):
            if self.isValid(idx):
                nextIdx = idx
                break
        return nextIdx

    def findPreviousIndex(self):
        """Similar to findNextIndex(), but we go backwards"""
        prevIdx = 0
        for idx in range(self.index-1, 0, -1):
            if self.isValid(idx):
                prevIdx = idx
                break
        return prevIdx
    
    def findDescendent(self, number):
        """
        find the descendent(s) of the cell 'number'
        """
        found = np.zeros( (1, 0), dtype='int32')
        #check if the cell has moved
        [ind] = np.nonzero(self.moves[:,0] == number)
        if ind.size == 1:
            found = np.array([self.moves[ind[0],1]])   
        #check if the cell has split
        [ind] = np.nonzero(self.splits[:,0] == number)
        if ind.size == 1:
            found = self.splits[ind[0],1:3]
        return found

    def findAncestor(self, number):
        """
        find the ancestor of the cell 'number'
        """
        found = np.zeros( (1, 0), dtype='int32')
        # check if the cell results from a movement
        [ind] = np.nonzero(self.moves[:,1] == number)
        if ind.size == 1:
            print "Ancestor from movement, row ", ind[0]
            found = np.array([self.moves[ind[0],0]])
        # check if the cell results from a split
        [ind1] = np.nonzero(self.splits[:,1] == number)
        [ind2] = np.nonzero(self.splits[:,2] == number)
        ind = np.concatenate((ind1, ind2))
        if ind.size==1:
            print "Ancestor from split, row ", ind[0]
            found = np.array([self.splits[ind[0],0]])
        return found
    
    def getNeighbors(self, seg, reference):
        """
        returns the indices of all cell labels in 'seg' except 'reference'
        """
        """
        labels = []
        for i in range(0,seg.shape[0]):
            for j in range(0,seg.shape[1]):
                for k in range(0,seg.shape[2]):
                    if (seg[i, j, k] != 0) and (not seg[i,j,k] in labels) and (seg[i,j,k]!=reference):
                        labels = np.append(labels,seg[i,j,k])
        """
        labels = np.unique(seg)
        for i in range(0,labels.size):
            if labels[i] == reference:
                labels[i] = 0
        labels = np.unique(labels)
        if(labels[0] == 0):
            labels = labels[1:]
        return np.int16(labels)
        
    
    def drawVolumeWithoutReferenceCell(self, figure, cseg, reference, color=(0,0,1), opacity=0.3):
        """
        draw all cells except the cell with label 'reference'
        """
        seg = np.copy(cseg)
        """
        # transform volume into binary volume
        for i in range(0,seg.shape[0]):
            for j in range(0,seg.shape[1]):
                for k in range(0,seg.shape[2]):
                    if seg[i,j,k] != 0:
                        if seg[i,j,k] in np.array(reference):
                            seg[i,j,k] = 0
                        else:
                            seg[i,j,k] = 1
        """
        if np.array(reference).size == 1:
            seg = np.multiply((seg != reference).astype(np.int32), (seg != 0).astype(np.int32))
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                      opacity = opacity, color=color, figure=figure)
        elif np.array(reference).size == 0:
            seg = (seg != 0).astype(np.int32)
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                      opacity = opacity, color=color, figure=figure)
        elif np.array(reference).size == 2:
            seg1 = np.multiply((seg != reference[0]).astype(np.int32), (seg != 0).astype(np.int32))
            seg2 = np.multiply((seg != reference[1]).astype(np.int32), (seg != 0).astype(np.int32))
            seg = np.add(seg1,seg2)
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                      opacity = opacity, color=color, figure=figure)
        else:    
            print "Jetzt haben wir ein Problem..."

    def drawReferenceCell(self, figure, cseg, reference, color = (0,1,0), opacity=0.4):
        """
        only draw the cell with label 'reference'
        """
        refArray = np.array(reference)
        if refArray.size == 1:
            seg = np.copy(cseg)
            
            seg = (seg == reference).astype(np.int32)
            """
            # transform volume into binary volume
            for i in range(0,seg.shape[0]):
                for j in range(0,seg.shape[1]):
                    for k in range(0,seg.shape[2]):
                        if seg[i,j,k] != 0:
                            if seg[i,j,k] in np.array(reference):
                                seg[i,j,k] = 1
                            else:
                                seg[i,j,k] = 0
            """
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                          opacity = opacity, color=color, figure=figure)
        if refArray.size == 2:
            seg = np.copy(cseg)
            seg = np.add((seg == reference[0]).astype(np.int32), (seg == reference[1]).astype(np.int32))
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                          opacity = opacity, color=color, figure=figure)

    def drawArrows(self, figure, bbox, number, partner):
        """
        draw an 'arrow' from the cell 'number' to 'partner'
        """
        #load the center of mass position
        n = np.array(number)
        p = np.array(partner)
        n.shape = (n.size, )
        p.shape = (p.size, )
        if p.size > 0 and n.size > 0:
                com1 = self.file1["features"][str(n[0])]["com"][:]
                com2 = self.file2["features"][str(p[0])]["com"][:]
                #write the cell label as text
                mlab.text3d(com1[2]-bbox[2]+1,com1[1]-bbox[1]+1,com1[0]-bbox[0]+1, str(number), color=(1,1,1), figure=figure)
                
                #plot a point where the current cell is
                mlab.points3d([com1[2]-bbox[2]+1],[com1[1]-bbox[1]+1],[com1[0]-bbox[0]+1],color=(0,0,1), figure=figure)
                
                #plot a line to the descendant's center
                mlab.plot3d([com1[2]-bbox[2]+1,com2[2]-bbox[2]+1],
                            [com1[1]-bbox[1]+1,com2[1]-bbox[1]+1],
                            [com1[0]-bbox[0]+1,com2[0]-bbox[0]+1],
                            tube_radius=0.2, color=(1,0,0), figure=figure)
                
                #plot a second line, if there is a split
                if p.size == 2:
                    com3 = self.file2["features"][str(p[1])]["com"][:]
                    mlab.plot3d([com1[2]-bbox[2]+1,com3[2]-bbox[2]+1],
                                [com1[1]-bbox[1]+1,com3[1]-bbox[1]+1],
                                [com1[0]-bbox[0]+1,com3[0]-bbox[0]+1],
                                tube_radius=0.2, color=(1,0,0), figure=figure)

    def drawImagePlane(self, figure, raw, colmap='jet', planeOrientation = 'x_axes'):
        """
        draw an image plane
        """
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(raw), figure=figure, colormap=colmap, plane_orientation = planeOrientation)

    def showCell(self, number, firstFrame=True):
        """
        Show cell with label 'number', its surrounding, descendants, etc.
        The second argument specifies whether the given cell label corresponds to a cell
        from the first of the two time frames (default) or to a cell from the second frame
        """
        
        # set the bounding box of the volume to be displayed
        if firstFrame:
            bbox = self.file1["features"][str(number)]["bbox"][:]
        else:
            bbox = self.file2["features"][str(number)]["bbox"][:]
        bbox[0] = bbox[0] - min(self.borderSize, bbox[0]) # make sure that bbox[0] >= 0 (no array bound violation)
        bbox[1] = bbox[1] - min(self.borderSize, bbox[1])
        bbox[2] = bbox[2] - min(self.borderSize, bbox[2])
        bbox[3] = bbox[3] + min(self.borderSize, self.shape[2]-bbox[3]) # make sure that bbox[3] <= self.shape[0] (no array bound violation)
        bbox[4] = bbox[4] + min(self.borderSize, self.shape[1]-bbox[4])
        bbox[5] = bbox[5] + min(self.borderSize, self.shape[0]-bbox[5])
        if firstFrame:
            partner = self.findDescendent(number)
            number1 = number
            number2 = partner
            partnerStr = "descendent(s) "
        else:
            partner = self.findAncestor(number)
            number1 = partner
            number2 = number
            partnerStr = "ancestor "
        
        # load the data
        seg1 = self.file1["segmentation"]["labels"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        seg2 = self.file2["segmentation"]["labels"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        raw1 = self.file1["raw"]["volume"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        raw2 = self.file2["raw"]["volume"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        
        print "Drawing cell number ",number," and its ", partnerStr ,partner
        t0 = time.time()
        
        #draw everything
        fig1 = mlab.figure(1, size=(500,450))
        mlab.clf(fig1)
        self.drawImagePlane(fig1, raw1, 'gray')
        self.drawImagePlane(fig1, raw2, 'copper', 'y_axes')
        self.drawVolumeWithoutReferenceCell(fig1, seg1, number1, (0,0,1),0.2)
        self.drawVolumeWithoutReferenceCell(fig1, seg2, number2, (0.2,0,0.8),0.2)
        self.drawReferenceCell(fig1, seg1, number1, (0.5,1,0), 0.4)
        self.drawReferenceCell(fig1, seg2, number2, (0.8,0.8,0),0.3)
        self.drawArrows(fig1, bbox, number1, number2)
        if firstFrame:
            allneighbors = self.getNeighbors(seg1, number1)
            for i in allneighbors:
                p = self.findDescendent(i)
                self.drawArrows(fig1, bbox, i, p)
        else:
            allneighbors = self.getNeighbors(seg2, number2)
            for i in allneighbors:
                p = self.findAncestor(i)
                self.drawArrows(fig1, bbox, p, i)
        t = time.time() - t0
        print "Time for drawing:",t

    def showNextCell(self):
        newIndex = self.findNextIndex()
        if newIndex > 0:
            self.index = newIndex 
            self.showCell(self.index)
        else:
            print "No valid cells with higher indices available"
        
    def showPreviousCell(self):
        newIndex = self.findPreviousIndex()
        if newIndex > 0:
            self.Index = newIndex
            self.showCell(self.Index)
        else:
            print "No valid cells with smaller indices available"
        
    def jumpToCell(self, number):
        if self.isValid(number):
            self.index = number
            self.showCell(self.index)
        else:
            print "User-defined index does not correspond to valid cell"


            
class dialog(HasTraits):
    """
    Small dialoge class using traits.ui. Now you can jump from cell to cell using a button click.
    """
    def __init__(self, timestep1, timestep2):
        self.t = training(timestep1,timestep2)
        self.t.index = 1
        self.t.showCell(1)
    index = Int(1)
    btnShowNext = Button()
    btnGood = Button()
    btnBad = Button()
    btnJump = Button()
    btnSave = Button()
    strFile = String('results.txt')
    
    labels = np.zeros([0,2])
    
    def _btnShowNext_fired(self):
        self.t.showNextCell()
        self.index = self.t.index

    def _btnGood_fired(self):
        self.labels = np.append(self.labels,[[self.t.index,1]],axis=0)
        self.t.showNextCell()
        self.index = self.t.index

    def _btnBad_fired(self):
        self.labels = np.append(self.labels,[[self.t.index,0]],axis=0)
        self.t.showNextCell()
        self.index = self.t.index

    def _btnJump_fired(self):
        self.t.showCell(self.index)
        self.t.index = self.index

    def _btnSave_fired(self):
        if self.strFile.__len__() != 0:
            np.savetxt(self.strFile,np.int16(self.labels), fmt="%i",delimiter="\t")
            print "Labels written to",self.strFile 
            
    view = View('index', 
                Item('btnJump', label='Jump to index', show_label=False ),
                Item('btnShowNext', label='Show next cell', show_label=False ),
                Item('btnGood', label='Label as "Good"', show_label=False ),
                Item('btnBad', label='Label as "Bad"', show_label=False ),
                'strFile',
                Item('btnSave', label='Save labels', show_label=False ),
                )



def show(file1,file2):
    d = dialog(file1,file2)
    d.configure_traits()
