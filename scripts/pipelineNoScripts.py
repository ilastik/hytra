"""
Run the full pipeline, configured by a config file, but without calling a series of other scripts.
"""

# pythonpath modification to make hytra available
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import logging
try:
    import commentjson as json
except ImportError:
    import json
import dpct
from subprocess import check_call
import configargparse as argparse
import hytra.core.ilastik_project_options
from hytra.core.jsongraph import JsonTrackingGraph
from hytra.core.ilastikhypothesesgraph import IlastikHypothesesGraph
from hytra.core.fieldofview import FieldOfView
from hytra.core.jsonmergerresolver import JsonMergerResolver

def convertToDict(unknown):
    indicesOfParameters = [i for i, p in enumerate(unknown) if p.startswith('--')]
    keys = [u.replace('--', '') for u in [unknown[i] for i in indicesOfParameters]]
    values = []
    for i in indicesOfParameters:
        if i + 1 > len(unknown) or unknown[i + 1].startswith('--'):
            values.append(True)
        else:
            values.append(unknown[i + 1])
    return dict(zip(keys, values))

def constructFov(shape, t0, t1, scale=[1, 1, 1]):
    [xshape, yshape, zshape] = shape
    [xscale, yscale, zscale] = scale

    fov = FieldOfView(t0, 0, 0, 0, t1, xscale * (xshape - 1), yscale * (yshape - 1),
                      zscale * (zshape - 1))
    return fov

def run_pipeline(options, unknown):
    """
    Run the complete tracking pipeline by invoking the different steps.
    Using the `do-SOMETHING` switches one can configure which parts of the pipeline are run.

    **Params:**

    * `options`: the options of the tracking script as returned from argparse
    * `unknown`: unknown parameters read from the config file, needed in case merger resolving is supposed to be run.

    """

    params = convertToDict(unknown)
    
    if options.do_extract_weights:
        logging.info("Extracting weights from ilastik project...")
        weights = hytra.core.ilastik_project_options.extractWeightDictFromIlastikProject(options.ilastik_tracking_project)
    else:
        with open(options.weight_filename, 'r') as f:
            weights = json.load(f)

    if options.do_create_graph:
        logging.info("Create hypotheses graph...")

        import hytra.core.probabilitygenerator as probabilitygenerator
        from hytra.core.ilastik_project_options import IlastikProjectOptions
        ilpOptions = IlastikProjectOptions()
        ilpOptions.labelImagePath = params['label-image-path']
        ilpOptions.labelImageFilename = params['label-image-file']
        ilpOptions.rawImagePath = params['raw-data-path']
        ilpOptions.rawImageFilename = params['raw-data-file']
        try:
            ilpOptions.rawImageAxes = params['raw-data-axes']
        except:
            ilpOptions.rawImageAxes = 'txyzc'

        ilpOptions.sizeFilter = [int(params['min-size']), 100000]

        if 'object-count-classifier-file' in params:
            ilpOptions.objectCountClassifierFilename = params['object-count-classifier-file']
        else:
            ilpOptions.objectCountClassifierFilename = options.ilastik_tracking_project

        withDivisions = 'without-divisions' not in params
        if withDivisions:
            if 'division-classifier-file' in params:
                ilpOptions.divisionClassifierFilename = params['division-classifier-file']
            else:
                ilpOptions.divisionClassifierFilename = options.ilastik_tracking_project
        else:
            ilpOptions.divisionClassifierFilename = None

        probGenerator = probabilitygenerator.IlpProbabilityGenerator(ilpOptions, 
                                              pluginPaths=['../hytra/plugins'],
                                              useMultiprocessing=False)

        # if time_range is not None:
        #     traxelstore.timeRange = time_range

        probGenerator.fillTraxels(usePgmlink=False)
        fieldOfView = constructFov(probGenerator.shape,
                                   probGenerator.timeRange[0],
                                   probGenerator.timeRange[1],
                                   [probGenerator.x_scale,
                                   probGenerator.y_scale,
                                   probGenerator.z_scale])

        hypotheses_graph = IlastikHypothesesGraph(
            probabilityGenerator=probGenerator,
            timeRange=probGenerator.timeRange,
            maxNumObjects=int(params['max-number-objects']),
            numNearestNeighbors=int(params['max-nearest-neighbors']),
            fieldOfView=fieldOfView,
            withDivisions=withDivisions,
            divisionThreshold=0.1
        )

        withTracklets = True
        if withTracklets:
            hypotheses_graph = hypotheses_graph.generateTrackletGraph()

        hypotheses_graph.insertEnergies()
        trackingGraph = hypotheses_graph.toTrackingGraph()
    else:
        trackingGraph = JsonTrackingGraph(model_filename=options.model_filename)

    if options.do_convexify:
        logging.info("Convexifying graph energies...")
        trackingGraph.convexifyCosts()

    # get model out of trackingGraph
    model = trackingGraph.model

    if options.do_tracking:
        logging.info("Run tracking...")
        result = dpct.trackFlowBased(model, weights)
        hytra.core.jsongraph.writeToFormattedJSON(options.result_filename, result)

        if hypotheses_graph:
            # insert the solution into the hypotheses graph and from that deduce the lineages
            hypotheses_graph.insertSolution(result)
            hypotheses_graph.computeLineage()

    if options.do_merger_resolving:
        logging.info("Run merger resolving")
        trackingGraph = JsonTrackingGraph(model=model, result=result)
        merger_resolver = JsonMergerResolver(
            trackingGraph,
            ilpOptions.labelImageFilename,
            ilpOptions.labelImagePath,
            params['out-label-image-file'],
            ilpOptions.rawImageFilename,
            ilpOptions.rawImagePath,
            ilpOptions.rawImageAxes,
            ['../hytra/plugins'],
            True)
        ilpOptions.labelImagePath = params['label-image-path']
        ilpOptions.labelImageFilename = params['label-image-file']
        ilpOptions.rawImagePath = params['raw-data-path']
        ilpOptions.rawImageFilename = params['raw-data-file']
        try:
            ilpOptions.rawImageAxes = params['raw-data-axes']
        except:
            ilpOptions.rawImageAxes = 'txyzc'
        merger_resolver.run(None,  None)

        ## resolved model and result are now here:
        # merger_resolver.model
        # merger_resolver.result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cell Tracking Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file', required=True)

    parser.add_argument("--do-ctc-groundtruth-conversion", dest='do_ctc_groundtruth_conversion', action='store_true', default=False)
    parser.add_argument("--do-ctc-raw-data-conversion", dest='do_ctc_raw_data_conversion', action='store_true', default=False)
    parser.add_argument("--do-ctc-segmentation-conversion", dest='do_ctc_segmentation_conversion', action='store_true', default=False)
    parser.add_argument("--do-train-transition-classifier", dest='do_train_transition_classifier', action='store_true', default=False)
    parser.add_argument("--do-extract-weights", dest='do_extract_weights', action='store_true', default=False)
    parser.add_argument("--do-create-graph", dest='do_create_graph', action='store_true', default=False)
    parser.add_argument("--do-convexify", dest='do_convexify', action='store_true', default=False)
    parser.add_argument("--do-tracking", dest='do_tracking', action='store_true', default=False)
    parser.add_argument("--do-merger-resolving", dest='do_merger_resolving', action='store_true', default=False)
    parser.add_argument("--export-format", dest='export_format', type=str, default=None,
                        help='Export format may be one of: "ilastikH5", "ctc", "labelimage", or None')
    parser.add_argument("--ilastik-tracking-project", dest='ilastik_tracking_project', required=True,
                        type=str, help='ilastik tracking project file that contains the chosen weights')
    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json graph description')
    parser.add_argument('--result-json-file', required=True, type=str, dest='result_filename',
                        help='Filename of the json file containing results')
    parser.add_argument('--weight-json-file', required=True, type=str, dest='weight_filename',
                        help='Filename of the weights stored in json')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    run_pipeline(options, unknown)
