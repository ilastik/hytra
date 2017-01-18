from __future__ import unicode_literals
import numpy as np
import time
import h5py
from mayavi import mlab


    
def drawVolumeWithoutReferenceCell(figure, cseg, reference, color=(0,0,1), opacity=0.3):
    """
    Draw all cells in volume 'cseg' except the cell with label 'reference' into 'figure'.
    'reference' is assumed to be a 1D np.array with size 1 or 2.
    """
    # create a copy of the volume
    seg = np.copy(cseg)
    
    # transform volume into binary volume and draw isosurfaces from it
    # consider the case that reference may have 0,1 or 2 entries
    if reference.size == 1:
        # one reference cell
        seg = np.multiply((seg != reference[0]).astype(np.int32), 
                          (seg != 0).astype(np.int32))
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                  opacity = opacity, color=color, figure=figure)
    elif reference.size == 0:
        # no reference cell
        seg = (seg != 0).astype(np.int32)
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                  opacity = opacity, color=color, figure=figure)
    elif reference.size == 2:
        # two reference cells
        seg1 = np.multiply((seg != reference[0]).astype(np.int32), 
                           (seg != 0).astype(np.int32))
        seg2 = np.multiply((seg != reference[1]).astype(np.int32), 
                           (seg != 0).astype(np.int32))
        seg = np.array(np.add(seg1,seg2) > 0, dtype = np.int)
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                  opacity = opacity, color=color, figure=figure)



def drawReferenceCell(figure, cseg, reference, color = (0,1,0), opacity=0.4):
    """
    Only draw the cell with label 'reference' in volume 'cseg' into figure.
    'reference' is assumed to be a 1D np.array with size 1 or 2.
    """
    if reference.size == 1:
        seg = np.copy(cseg)
        # transform volume into binary volume
        seg = (seg == reference[0]).astype(np.int32)
        
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                  opacity = opacity, color=color, figure=figure)
    elif reference.size == 2:
        seg = np.copy(cseg)
        seg = np.add((seg == reference[0]).astype(np.int32), 
                     (seg == reference[1]).astype(np.int32))
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(seg), contours=[0.5], 
                                  opacity = opacity, color=color, figure=figure)

def drawArrows(file1, file2, figure, bbox, index, descendant):
    """
    Draw an 'arrow' from the cell 'index' to 'descendant'
    'descendant' is assumed to be a 1D np.array with size 0, 1 or 2.
    """
    #load the center of mass position
    if descendant.size > 0 and descendant[0] != 0:
        com1 = file1["features"][str(index[0])]["com"][:]
        com2 = file2["features"][str(descendant[0])]["com"][:]
        
        #write the cell label as text
        mlab.text3d(com1[2]-bbox[2]+1,com1[1]-bbox[1]+1,com1[0]-bbox[0]+1, str(index), color=(1,1,1), figure=figure)
        
        #plot a point where the current cell is
        mlab.points3d([com1[2]-bbox[2]+1],[com1[1]-bbox[1]+1],[com1[0]-bbox[0]+1],color=(0,0,1), figure=figure)
                
        #plot a line to the descendant's center
        mlab.plot3d([com1[2]-bbox[2]+1,com2[2]-bbox[2]+1],
                    [com1[1]-bbox[1]+1,com2[1]-bbox[1]+1],
                    [com1[0]-bbox[0]+1,com2[0]-bbox[0]+1],
                    tube_radius=0.2, color=(1,0,0), figure=figure)
        
        #plot a second line, if there is a split
        if descendant.size == 2:
            com3 = file2["features"][str(descendant[1])]["com"][:]
            mlab.plot3d([com1[2]-bbox[2]+1,com3[2]-bbox[2]+1],
                        [com1[1]-bbox[1]+1,com3[1]-bbox[1]+1],
                        [com1[0]-bbox[0]+1,com3[0]-bbox[0]+1],
                        tube_radius=0.2, color=(1,0,0), figure=figure)

def drawLabels(figure, handle, vol, bbox):
    """
    Draw all labels in the volume 'vol' at the center of mass position of the object
    into the figure 'figure'. Therefore, the bounding box 'bbox' coordinates of 'vol'
    must be specified.
    """
    # find all labels
    labels = np.unique(vol)
    if(labels[0] == 0):
        # remove background label
        labels = labels[1:]

    #write the cell label as text
    for i in labels:
        com = handle["features"][str(i)]["com"][:]
        mlab.text3d(com[2]-bbox[2]+1,com[1]-bbox[1]+1,com[0]-bbox[0]+1, str(i), color=(1,1,1), figure=figure)
        

def drawImagePlane(figure, raw, colmap='jet', planeOrientation = 'z_axes'):
    """
    Draw an image plane of the volume data 'raw' into figure 'figure'.
    Colormap can be specified by setting 'colormap', starting plane orientation
    is set in 'planeOrientation'.
    """
    ipw = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(raw), figure=figure, colormap=colmap, plane_orientation = planeOrientation,vmin = raw.min(), vmax=raw.max())
    ipw.ipw.color_map.output_format = 'rgb'
    #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(raw), figure=figure, colormap=colmap, vmin = 0, vmax = 255, plane_orientation = planeOrientation)


def draw2DView(figure, raw, seg, index, color=(1,0,0)):
    """
    Draw a 2D maximum intensity projection of the raw data and create 
    a segmentation overlay
    """
    raw2d = np.max(raw, axis = 2)
    mlab.imshow(raw2d, colormap='gray', figure=figure)
    sother = np.multiply( (seg != 0).astype(np.int32),(seg != index).astype(np.int32) )
    ##max1 = np.max(np.where(np.max(np.max((seg == index).astype(np.int32), axis = 0), axis = 0)!=0))+1
    ##min1 = np.min(np.where(np.max(np.max((seg == index).astype(np.int32), axis = 0), axis = 0)!=0))
    ##max2 = np.max(np.where(np.max(np.max((seg == index).astype(np.int32), axis = 0), axis = 1)!=0))+1
    ##min2 = np.min(np.where(np.max(np.max((seg == index).astype(np.int32), axis = 0), axis = 1)!=0))
    ##d1 = int(min(max1-min1-15,0))
    ##d2 = int(min(max2-min2-15,0))
    ##min1 += d1
    ##max1 -= d1
    ##min2 += d2
    ##max2 -= d2
    ##sother = (seg == index).astype(np.int32)
    ##seg2dother = np.max(sother, axis = 0)
    ##seg2dother[min2:max2,min1:max1]=1
    sindex = (seg == index).astype(np.int32)
    seg2dother = np.max(sother, axis = 2)
    seg2dindex = np.max(sindex, axis = 2)
    mlab.contour_surf(seg2dother, color = (0,0,1), contours = 2, warp_scale = 0, figure=figure)
    mlab.contour_surf(seg2dindex, color = color, contours = 2, warp_scale = 0, figure=figure)
    #figure.scene.z_plus_view()


def drawScene(filename, index, borderSize=25):
    """
    Draw a complete scene of cell 'index' from file 'filename', including segmented
    shape, surrounding objects and image plane.
    """
    f = h5py.File(filename)
    # create an array containing 'index'
    index = np.array([index], dtype=np.int32)
    
    # set the bounding box of the volume to be displayed
    bbox = f["features"][str(index[0])]["bbox"][:]
    
    #enlarge the bounding box
    # make sure that bbox[0..2] >= 0 (no array bound violation)
    bbox[0] = bbox[0] - min(borderSize, bbox[0]) 
    bbox[1] = bbox[1] - min(borderSize, bbox[1])
    bbox[2] = bbox[2] - min(borderSize, bbox[2])
    # make sure that bbox[3..5] <= self.shape[0] (no array bound violation)
    bbox[3] = bbox[3] + min(borderSize, f['raw']['volume'].shape[2]-bbox[3]) 
    bbox[4] = bbox[4] + min(borderSize, f['raw']['volume'].shape[1]-bbox[4])
    bbox[5] = bbox[5] + min(borderSize, f['raw']['volume'].shape[0]-bbox[5])
    
    # load the data
    seg1 = f["segmentation"]["labels"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
    raw = f["raw"]["volume"][bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]]
        
    #draw everything
    fig1 = mlab.figure(1, size=(500,450))
    mlab.clf(fig1)
    drawImagePlane(fig1, raw, 'gist_ncar') 
    drawVolumeWithoutReferenceCell(fig1, seg1, index, (0,0,1),0.2)
    drawReferenceCell(fig1, seg1, index, (0.5,1,0), 0.4)
    
    fig2 = mlab.figure(2, size=(500,450))
    mlab.clf(fig2)
    
    draw2DView(fig2, raw[20:-20,:,:], seg1[20:-20,:,:], index)
    

