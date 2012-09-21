import os.path
import sys

VERBOSE = True

# try to import python wrapper from project source
module_path_in_source = 'tracking/python/'
if(os.path.exists(module_path_in_source) or os.path.exists('../' + module_path_in_source)):
   sys.path.append(module_path_in_source)
   sys.path.append('../' + module_path_in_source)
try:
   import ctracking as _ct

   from ctracking import FieldOfView

   from ctracking import ComLocator
   from ctracking import IntmaxposLocator
   from ctracking import Traxel as cTraxel
   from ctracking import Traxels as cTraxels
   from ctracking import TraxelStore

   from ctracking import Event
   from ctracking import EventType
   from ctracking import EventVector
   #from ctracking import FixedCostTracking
   #from ctracking import ShortestDistanceTracking
   #from ctracking import CellnessTracking
   #from ctracking import BotTracking
   from ctracking import MrfTracking
   #from ctracking import KanadeTracking   
   
except ImportError as error:
   if VERBOSE:
      print "-! ctracking module not found or incompatible. Related functionality not available.: %s" % error 
      
