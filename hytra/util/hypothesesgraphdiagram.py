from hytra.core.hypothesesgraph import HypothesesGraph
import numpy as np
import matplotlib.pyplot as plt
import colorsys


class HypothesesGraphDiagram(object):
    def __init__():
        pass
    
    
    '''
    Save json tracking graph diagram for debugging purposes
    '''
    @staticmethod
    def saveJsonGraphDiagram(model, results=None, timeRange=(0, 10), width=4000, height=2000, radius=20, fileName='JsonGraph.png'):
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
        def get_colors(num_colors):
            colors=[]
            for i in np.arange(0., 360., 360. / num_colors):
                hue = i/360.
                lightness = (50 + np.random.rand() * 10)/100.
                saturation = (90 + np.random.rand() * 10)/100.
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
            return colors
        
        colors = get_colors(5)
 
        # Get active arcs map
        # Add code to obtain active arcs in this section
        
        if results:
            # Get result dict by id
            detResultsByUuid = {}
            for detection in results['detectionResults']:
                detResultsByUuid[detection['id']] = detection['value']
                
            # Get result dict by id
            linkResultsByUuid = {}
            for link in results['linkingResults']:
                linkResultsByUuid[(link['src'], link['dest'])] = link['value']
        
        # Node by uuid 
        nodeByUuid = {}
        for node in model['segmentationHypotheses']:
            uuid = node['id']
            nodeByUuid[uuid] = node
         
        # Draw nodes
        nodeCoordsMap = {}
    
        # Draw nodes 
        for node in model['segmentationHypotheses']:
            time = node['nid'][0]
            id = node['nid'][1]
            uuid = node['id']
            
            # Node is inactive
            if results and not (uuid in detResultsByUuid and detResultsByUuid[uuid] > 0):
                continue
                        
            # Verify that node is within time ranges and plot nodes
            if time >= timeRange[0] and time < timeRange[1]:                         
                nodeCoordsMap[(time, id)] = ( (time-timeRange[0]+1)*columnWidth, id*rowHeight)
                rowPos[time] += rowHeight
                 
                # Set the color of active nodes
                faceColor = colors[1]
                nodeLabel = str('')
                edgeColor = 'k'
                 
                # Display label and color from node value                
                if results:
                    #faceColor = colors[detResultsByUuid[uuid]] 
                    nodeLabel = str(detResultsByUuid[uuid])
                
                # Set color if node has appearance/disappearance features
                if 'appearanceFeatures' in node:
                    faceColor = colors[2] 
                elif 'disappearanceFeatures' in node:
                    faceColor = colors[3]
                    
                circle=plt.Circle(nodeCoordsMap[(time, id)], radius=radius, edgecolor=edgeColor, facecolor=faceColor, fill=True, zorder=2)
                ax.add_patch(circle)
                        
                ax.annotate(nodeLabel,
                xy=(0, 0),  
                xytext=nodeCoordsMap[(time, id)],
                horizontalalignment='center',
                verticalalignment='center',
                zorder=10)           
         
        # Draw arcs           
        for link in model['linkingHypotheses']:
            uuidSource = link['src']
            uuidTarget = link['dest']
            
            if results and not ((uuidSource, uuidTarget) in linkResultsByUuid and linkResultsByUuid[(uuidSource, uuidTarget)] > 0) :
                continue
            
            nodeSource = nodeByUuid[uuidSource]
            nodeTarget = nodeByUuid[uuidTarget]
            
            sourceNode = nodeSource['nid']
            targetNode = nodeTarget['nid']
             
            sourceTime = sourceNode[0]
            targetTime = targetNode[0]
             
            # Verify that the arc nodes are found within the time range
            if sourceTime >= timeRange[0] and sourceTime < timeRange[1] and targetTime >= timeRange[0] and targetTime < timeRange[1]:  
                color = 'k'
                linestyle = 'solid'
                linewidth = 1.0               
 
                xArcCoords = [ nodeCoordsMap[sourceNode][0], nodeCoordsMap[targetNode][0] ]
                yArcCoords = [ nodeCoordsMap[sourceNode][1], nodeCoordsMap[targetNode][1] ]
 
                line=plt.Line2D( xArcCoords, yArcCoords, linewidth=linewidth, color=color, linestyle=linestyle, zorder=1)
                ax.add_line(line)            
 
        # Save figure
        plt.axis('off')
        plt.savefig(fileName)
        
        

    '''
    Save hypotheses graph (networkx digraph) diagram for debugging purposes
    '''
    @staticmethod
    def saveHypothesesGraphDiagram(hypothesesGraph, timeRange=(0, 10), width=4000, height=2000, radius=20, withNodeValue=True, withArcValue=True, withArcFeatures=False, withUuid=False, fileName='HypothesesGraph.png', csvFileName=None):         
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
        def get_colors(num_colors):
            colors=[]
            for i in np.arange(0., 360., 360. / num_colors):
                hue = i/360.
                lightness = (50 + np.random.rand() * 10)/100.
                saturation = (90 + np.random.rand() * 10)/100.
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
            return colors
        
        colors = get_colors(5)
 
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
            
 
        # Save figure
        plt.axis('off')
        plt.savefig(fileName)



    
