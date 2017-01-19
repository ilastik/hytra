#!/usr/bin/python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from builtins import range
from past.utils import old_div
import operator
from h5py import h5s, h5d, h5f, h5g
import numpy as np
from functools import reduce

def readDataSet( gid, name, dt=np.int32 ):
  did = h5d.open( gid, name )
  space_id = did.get_space()
  dims = space_id.get_simple_extent_dims()
  memsp_id = h5s.create_simple( dims )
  data = np.zeros(dims, dtype=dt)
  did.read(memsp_id, space_id, data)
  return data

def getIntensity( fileId, cellNumber ):
  intensities = readDataSet(fileId, '/features/%d/intensity' % cellNumber)
  return intensities[0] 

def getColorRankIndex( fileId, cellNumber, intensWindow, minIndex, isNewCell=False,sortByIntensities=False ):  
  """Return the color string and the rank of a cell"""
  if len(intensWindow) == 2 or sortByIntensities:
    intensity = getIntensity(fileId, cellNumber)
  else:
    intensity = 0
  if sortByIntensities:
    rank = -intensity
  else:
    rank = cellNumber
  hue = 0.0
  if len(intensWindow)==2:
    addendum = ', style="filled"'
    index = (1-minIndex)*(intensity - intensWindow[0]) / (intensWindow[1] - intensWindow[0]) + minIndex
    if index > 1.0:
      index = 1.0
    if index < minIndex:
      index = minIndex
    if isNewCell:
      saturation = index
      value = 1.0
    else:
      saturation = 0.0
      value = 1 - index
  else:
    if isNewCell:
      saturation = 1.0
    else:
      saturation = 0.0
    value = 0.0
    addendum = ""
    index = minIndex
  colorString = ('color="%f %f %f"' % (hue, saturation, value)) + addendum
  return (colorString, rank, index) 
 
def visualizeGraph( fileStr, fileNrs, dotNameLinear, penWidthNodes=2, penWidthEdges=2, intensWindow=[], \
                    sortByIntensities=False ):
  """Use Graphviz for visualizing the lineage tree implicitly contained in the files 
  fileStr % fn (for all fn in fileNrs). The generated DOT instructions will be stored in the file 
  dotNameLinear (for the linear layout) and the total number of cells is returned.
  Scaling factors for the pen width (line width) may be specified as well for nodes and edges. 
  Furthermore an intensity window can be provided that adjusts the fill colors of
  nodes as well as the line width of edges: if intensWindow == [], all nodes are
  printed the same way, but if intensWindow == [minIntens, maxIntens], the intensity
  range [minIntens, maxIntens] is mapped to the value range of 0.2 - 1 in the HSV
  color model. If sortByIntensities == True, the nodes are sorted by the average
  intensities of the cells in descending order."""
  minIndex = 0.05
  if len(intensWindow) != 0:
    if (len(intensWindow) != 2) or (intensWindow[0] >= intensWindow[1]):
      raise RuntimeError("If intensWindow non-empty, it must contain two values (with the first " \
        + "being smaller")
  outLin = open(dotNameLinear,'w')
  print('graph G {', file=outLin)
  # we keep a dictionary of all currently living cells, which maps the integer number of this
  # cell to the tuple consisting of the index of the node by which this cell is represented 
  # in the graph, plus the list of intensities from which the final pen width can be
  # determined
  livingCells = dict()
  # We represent all nodes by a jagged array, with all nodes on the same level corresponding to a 
  # subarray. The first node is always the time line node. Newly appearing nodes are plotted in
  # red, while disappearing nodes are marked with a cross.
  fstr = fileStr % fileNrs[0]
  fid = h5f.open( fstr, h5f.ACC_RDONLY )
  print('subgraph %d {\nrank=same;' % fileNrs[0], file=outLin)
  # list of all edges which are added to the graph at the end
  edgesLin = []
  print('%d [color=white, shape=circle];' % fileNrs[0], file=outLin)
  labcontent = readDataSet(fid, '/features/labelcontent', np.uint16)
  cellIdx = 1
  allCellStrs = []
  for idx in range( len(labcontent) ):
    if labcontent[idx]:
      cellStr = 'C%d_%d' % (fileNrs[0], idx+1)
      cii = getColorRankIndex(fid, idx+1, intensWindow, minIndex, False, \
            sortByIntensities)
      allCellStrs.append( (cellStr, cii[0], cii[1], cii[2]) )
      livingCells[ idx+1 ] = (cellStr, [cii[2]])
      cellIdx = cellIdx + 1
  fid.close()
  del fid
  allCellStrs.sort(key=lambda x : x[2])
  for cs in allCellStrs:
    print(cs[0] + (' [shape=circle, penwidth=%f, ' % (cs[3]*penWidthNodes)) + cs[1] + '];', file=outLin) 
  print('}', file=outLin)
  fnPrev = fileNrs[0]
  for fn in fileNrs[1:]:
    print('Processing file no. %d' % fn)
    print('subgraph %d {\nrank=same;' % fn, file=outLin)
    print('%d [color=white, shape=circle];' % fn, file=outLin)
    edgesLin.append('%d -- %d [penwidth=%f];' % (fnPrev, fn, 0.1*penWidthEdges) )
    fid = h5f.open(fileStr % fn, h5f.ACC_RDONLY)
    gid = h5g.open(fid,'/tracking')
    newLivingCells = dict()
    allCellStrs = []
    if 'Moves' in gid:
      moves = readDataSet(gid, 'Moves')
      nMoves = moves.shape[0]
      for iM in range(nMoves):
        newLivingCells[ moves[iM, 1] ] = livingCells[ moves[iM, 0] ]
        cii = getColorRankIndex(fid, moves[iM, 1], intensWindow, minIndex, False, sortByIntensities)
        newLivingCells[ moves[iM, 1] ][1].append(cii[2])
    if 'Splits' in gid:
      splits = readDataSet(gid, 'Splits')
      nSplits = splits.shape[0]
      for iS in range(nSplits):
        (motherStr, motherInts) = livingCells[ splits[iS, 0] ]
        cellStr1 = 'C%d_%d' % (fn, splits[iS, 1])
        cellStr2 = 'C%d_%d' % (fn, splits[iS, 2])
        cii1 = getColorRankIndex(fid, splits[iS, 1], intensWindow, minIndex, False, sortByIntensities)
        cii2 = getColorRankIndex(fid, splits[iS, 2], intensWindow, minIndex, False, sortByIntensities)
        allCellStrs.append( (cellStr1, cii1[0], cii1[1], cii1[2]) )
        allCellStrs.append( (cellStr2, cii2[0], cii2[1], cii2[2]) )
        newLivingCells[ splits[iS, 1] ] = (cellStr1, [cii1[2]])
        ints1 = motherInts + [cii1[2]]
        meanInt1 = old_div(reduce(operator.add, ints1), len(ints1))
        edgesLin.append(motherStr + ' -- ' + cellStr1 + ' [penwidth=%f];' % (meanInt1*penWidthEdges))
        newLivingCells[ splits[iS, 2] ] = (cellStr2, [cii2[2]])
        ints2 = motherInts + [cii2[2]]
        meanInt2 = old_div(reduce(operator.add, ints2), len(ints2))
        edgesLin.append(motherStr + ' -- ' + cellStr2 + ' [penwidth=%f];' % (meanInt2*penWidthEdges))
        cellIdx = cellIdx + 2
    if 'Appearances' in gid:
      apps = readDataSet(gid, 'Appearances')
      nApps = apps.shape[0]
      for iA in range(nApps):
        cellStr = 'C%d_%d' % (fn, apps[iA])
        cii = getColorRankIndex(fid, apps[iA], intensWindow, minIndex, True, sortByIntensities)
        allCellStrs.append( (cellStr, cii[0], cii[1], cii[2]) )
        newLivingCells[ apps[iA] ] = (cellStr, [cii[2]])
        cellIdx = cellIdx + 1
    if 'Disappearances' in gid:
      disapps = readDataSet(gid, 'Disappearances')
      nDisapps = disapps.shape[0]
      for iDA in range(nDisapps):
        cellStr = 'D%d_%d' % (fn, disapps[iDA])
        print(cellStr + ' [shape=point, penwidth=%f];' % (minIndex*penWidthNodes), file=outLin)
        (motherStr, motherInts) = livingCells[ disapps[iDA] ]
        meanInt = old_div(reduce(operator.add, motherInts), len(motherInts))
        edgesLin.append(motherStr + ' -- ' + cellStr + ' [penwidth=%f];' % (meanInt*penWidthEdges) )
        cellIdx = cellIdx + 1
    fid.close()
    del fid
    for cs in allCellStrs:
      print(cs[0] + (' [shape=circle, penwidth=%f, ' % (cs[3]*penWidthNodes)) + cs[1] + '];', file=outLin) 
    print('}\n', file=outLin)
    livingCells = newLivingCells
    fnPrev = fn
  print('subgraph end {\nrank=same;', file=outLin)
  for k in list(livingCells.keys()):
    cellStr = 'L%d' % k
    print(cellStr + ' [shape=none, penwidth=%f];' % (2*minIndex*penWidthNodes), file=outLin)
    (motherStr, motherInts) = livingCells[k]
    meanInt = old_div(reduce(operator.add, motherInts),len(motherInts))
    edgesLin.append(motherStr + ' -- ' + cellStr + ' [penwidth=%f];' % (0.2*penWidthEdges) )
  print('}\n', file=outLin)
  for e in edgesLin:
    print(e, file=outLin)
  print('}', file=outLin) 
  outLin.close()
  return (cellIdx-1)

# create a series of dot files in order to study the asymptotic run-time of graph plotting
# programs
if __name__ == '__main__':
  import pickle
  masterdir = '/home/fkaster/data_animal/gc/version2/';
  nCellsFileStr = masterdir + 'evaluation/nCells2.txt'
  fileStr = masterdir + '%04d_z-interpolation=2.h5'
  endNrs = list(range(10, 61, 10))
  intensWindow = (200, 2000)
  rangeStart = 0
  nCells = dict()
  for en in endNrs:
    fileNrs = list(range(rangeStart, en))  
    dotNameLin = masterdir + 'evaluation/hciLineageTree_%d.dot' % en
    currNCells = visualizeGraph(fileStr, fileNrs, dotNameLin, 10, 50, intensWindow, True) 
    nCells[en] = currNCells
  fid = open(nCellsFileStr,'w')
  pickle.dump(nCells, fid)
  fid.close()
