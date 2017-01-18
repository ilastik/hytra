from __future__ import unicode_literals
from enthought.mayavi import mlab

def cutPlanes( volume, colormap='gist_ncar' ):
    '''Display a 3D volume of scalars with two cut planes.

    volume: a three dimensional array of scalars
    '''
    
    scalarField = mlab.pipeline.scalar_field( volume )
    
    mlab.pipeline.image_plane_widget(scalarField,
                            plane_orientation='z_axes',
                            slice_index=10,
                            colormap = colormap
                                     )
    mlab.pipeline.image_plane_widget(scalarField,
                            plane_orientation='y_axes',
                            slice_index=10,
                            colormap = colormap
                                     )
    mlab.outline()
    mlab.axes()
    mlab.colorbar(orientation='vertical')
