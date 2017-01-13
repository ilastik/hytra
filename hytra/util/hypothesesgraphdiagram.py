from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
from hytra.core.hypothesesgraph import HypothesesGraph
import numpy as np
import matplotlib.pyplot as plt
import colorsys


class HypothesesGraphDiagram(object):
    def __init__(self, hypothesesGraph, timeRange=(0, 10), width=4000, height=2000, radius=20, withNodeValue=True, withArcValue=True, withArcFeatures=False, withUuid=False, fileName='HypothesesGraph.png', csvFileName=None):         
        # Initalize and configure plt plot
        plt.clf()
        fig=plt.figure(1)
        dpi = fig.get_dpi()
        fig.set_size_inches(width/float(dpi), height/float(dpi))  
        plt.axis([0,width,0,height])
        ax=fig.add_subplot(1,1,1)
         
        # Compute dimensions
        columnWidth = width/float(len( range(timeRange[0], timeRange[1]) )+1)
        rowHeight = 3*radius
          
        # Compute row position and add time labels
        rowPos = {} 
        for time in range(timeRange[0], timeRange[1]):
            rowPos[time] = radius*4
              
            ax.annotate(str(time),
                xy=(0, 0),  
                xytext=((time-timeRange[0]+1)*columnWidth,radius), # fraction, fraction
                horizontalalignment='center',
                verticalalignment='center',
                zorder=10)              
 
        # Get color list
        colors = self._get_colors(5)
 
        # Get active arcs map
        # Add code to obtain active arcs in this section
         
        # Draw nodes
        nodeCoordsMap = {}
         
        # Draw nodes 
        for node in hypothesesGraph.nodes_iter():
            time = node[0]
            id = node[1]
            
            # Verify that node is within time ranges and plot nodes
            if time >= timeRange[0] and time < timeRange[1]:  
                nodeCoordsMap[node] = ( (time-timeRange[0]+1)*columnWidth, id*rowHeight)
                rowPos[time] += rowHeight
                
                # Set the color of active nodes
                faceColor = colors[1]
                nodeLabel = str('')#id)
                edgeColor = 'k'
                
                # Display label and color from node value
                if withNodeValue and 'value' in hypothesesGraph.node[node]:
                    faceColor = colors[hypothesesGraph.node[node]['value']] 
                    nodeLabel = str(hypothesesGraph.node[node]['value'])
                
                # Display uuid on node labels instead    
                if withUuid and 'id' in hypothesesGraph.node[node]:
                    nodeLabel = str(hypothesesGraph.node[node]['id'])
                     
                    
                circle=plt.Circle(nodeCoordsMap[node], radius=radius, edgecolor=edgeColor, facecolor=faceColor, fill=True, zorder=2)
                ax.add_patch(circle)
                        
                ax.annotate(nodeLabel,
                xy=(0, 0),  
                xytext=nodeCoordsMap[node],
                horizontalalignment='center',
                verticalalignment='center',
                zorder=10)           
         
        # Draw arcs           
        for arc in hypothesesGraph.edges_iter():
            sourceNode = arc[0]
            targetNode = arc[1]
            
            sourceTime = sourceNode[0]
            targetTime = targetNode[0]
            
            # Verify that the arc nodes are found within the time range
            if sourceTime >= timeRange[0] and sourceTime < timeRange[1] and targetTime >= timeRange[0] and targetTime < timeRange[1]:  
                color = 'k'
                linestyle = 'solid'
                linewidth = 0.1
                
                if withArcValue and 'value' in hypothesesGraph.edge[sourceNode][targetNode]:
                    linewidth += 1.5*float( hypothesesGraph.edge[sourceNode][targetNode]['value'] )
                    if hypothesesGraph.edge[sourceNode][targetNode]['value'] == 0:
                        color = 'g'
                elif withArcFeatures and 'features' in hypothesesGraph.edge[sourceNode][targetNode]:
                    linewidth += 10.0*float( hypothesesGraph.edge[sourceNode][targetNode]['features'][0][0] ) 
                else:
                    linewidth=1.0                 

                xArcCoords = [ nodeCoordsMap[sourceNode][0], nodeCoordsMap[targetNode][0] ]
                yArcCoords = [ nodeCoordsMap[sourceNode][1], nodeCoordsMap[targetNode][1] ]

                line=plt.Line2D( xArcCoords, yArcCoords, linewidth=linewidth, color=color, linestyle=linestyle, zorder=1)
                ax.add_line(line)  

        # Load node list of a walk from a csv file. The file must contain a 'time' column and a 'id' column
        if csvFileName: 
            # Csv file must contain 'time' and 'id' columns      
            data = np.genfromtxt(csvFileName,delimiter=',', dtype=int, names=True)
     
            arcsWeight = {}
            
            # Generate the arc weights and plot the nodes 
            for i in range(len(data['time'])-1):
                sourceNode = (data['time'][i], data['id'][i])
                targetNode = (data['time'][i+1], data['id'][i+1])
                
                sourceTime = sourceNode[0] 
                targetTime = targetNode[0] 
                                         
                if sourceTime >= timeRange[0] and sourceTime < timeRange[1] and  targetTime >= timeRange[0] and targetTime < timeRange[1]:
                    if targetTime - sourceTime == 1:
                        arc=(sourceNode,targetNode)
                        
                        # Set arc weight based on the number of times that we have walked this path 
                        if arc in arcsWeight:
                            arcsWeight[arc] += 0.7
                        else: 
                            arcsWeight[arc] = 0.3    
                    
                    circle=plt.Circle(nodeCoordsMap[sourceNode], radius=radius, edgecolor='k', facecolor='r', fill=True, zorder=9)
                    ax.add_patch(circle)
                     
            # Plot the arcs     
            for arc in arcsWeight.keys():        
                sourceNode = arc[0]
                targetNode = arc[1]
                 
                if sourceNode[0] >= timeRange[0] and sourceNode[0] < timeRange[1] and targetNode[0] >= timeRange[0] and targetNode[0] < timeRange[1]: 
                    color = 'k'
                    linestyle = 'solid'
                    linewidth = arcsWeight[arc]
                 
                    coords = [ [nodeCoordsMap[sourceNode][0],nodeCoordsMap[targetNode][0]], [nodeCoordsMap[sourceNode][1],nodeCoordsMap[targetNode][1]] ]
                 
                    line=plt.Line2D( coords[0], coords[1], linewidth=linewidth, color=color, linestyle=linestyle, zorder=8)
                    ax.add_line(line)            
 
        # Save figure
        plt.axis('off')
        plt.savefig(fileName)


    # Generate color list
    def _get_colors(self, num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors
    
