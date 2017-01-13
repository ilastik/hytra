# pythonpath modification to make hytra available 
# for import without requiring it to be installed
from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import configargparse as argparse
import hytra.core.jsongraph
import hytra.core.ilastik_project_options

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the weights from an ilastik tracking project.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file')
    parser.add_argument('--ilastik-tracking-project', required=True, type=str, dest='ilpFilename',
                        help='Filename of the ilastik tracking project')
    parser.add_argument('--weight-json-file', type=str, dest='out', required=True, help='Filename of the resulting weights JSON file')
    parser.add_argument('--param-path', type=str, dest='param_path', default='/ConservationTracking/Parameters',
                        help='Path inside ilastik project that stores the parameters. Old ilastik projects used "/ConservationTracking/Parameters/0000"')

    args, _ = parser.parse_known_args()

    weightsDict = hytra.core.ilastik_project_options.extractWeightDictFromIlastikProject(args.ilpFilename, args.param_path)
    hytra.core.jsongraph.writeToFormattedJSON(args.out, weightsDict)
