from __future__ import print_function
from __future__ import unicode_literals
import os.path
import sys

VERBOSE = True

# try to import python wrapper from project source
module_path_in_source = 'tracking/build/python/'
if(os.path.exists(module_path_in_source) or os.path.exists('../' + module_path_in_source)):
   sys.path.append(module_path_in_source)
   sys.path.append('../' + module_path_in_source)
try:
   import pgmlink as _ct

   from pgmlink import FieldOfView

   from pgmlink import ComLocator
   from pgmlink import IntmaxposLocator
   from pgmlink import Traxel as cTraxel
   from pgmlink import Traxels as cTraxels
   from pgmlink import TraxelStore

   from pgmlink import Event
   from pgmlink import EventType
   from pgmlink import EventVector
   #from pgmlink import FixedCostTracking
   #from pgmlink import ShortestDistanceTracking
   #from pgmlink import CellnessTracking
   #from pgmlink import BotTracking
   from pgmlink import ChaingraphTracking
   from pgmlink import ConsTracking
   #from pgmlink import KanadeTracking
   from pgmlink import VectorOfInt
   from pgmlink import Traxel
   
except ImportError as error:
   if VERBOSE:
      print("-! pgmlink module not found or incompatible. Related functionality not available.: %s" % error) 
      
