"""
Run the full pipeline, configured by a config file
"""

# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import logging
from subprocess import check_call
import configargparse as argparse

def run_pipeline(options, unknown):
    """
    Run the complete tracking pipeline by invoking the scripts as subprocesses.
    Using the `do-SOMETHING` switches one can configure which parts of the pipeline are run.

    **Params:**

    * `options`: the options of the tracking script as returned from argparse
    * `unknown`: unknown parameters read from the config file, needed in case merger resolving is supposed to be run.

    """

    if options.do_ctc_groundtruth_conversion:
        logging.info("Convert CTC groundtruth to our format...")
        check_call(["python", os.path.abspath("ctc/ctc_gt_to_hdf5.py"), "--config", options.config_file])

    if options.do_ctc_raw_data_conversion:
        logging.info("Convert CTC raw data to HDF5...")
        check_call(["python", os.path.abspath("ctc/stack_to_h5.py"), "--config", options.config_file])

    if options.do_ctc_segmentation_conversion:
        logging.info("Convert CTC segmentation to HDF5...")
        check_call(["python", os.path.abspath("ctc/segmentation_to_hdf5.py"), "--config", options.config_file])

    if options.do_train_transition_classifier:
        logging.info("Train transition classifier...")
        check_call(["python", os.path.abspath("train_transition_classifier.py"), "--config", options.config_file])

    if options.do_extract_weights:
        logging.info("Extracting weights from ilastik project...")
        check_call(["python", os.path.abspath("tracking_ilp_to_weights.py"), "--config", options.config_file])

    if options.do_create_graph:
        logging.info("Create hypotheses graph...")
        check_call(["python", os.path.abspath("hypotheses_graph_to_json.py"), "--config", options.config_file])

    if options.do_convexify:
        logging.info("Convexifying graph energies...")
        check_call(["python", os.path.abspath("convexify_costs.py"), "--config", options.config_file])

    if options.do_tracking:
        logging.info("Run tracking...")

        if options.tracking_executable is not None:
            check_call([options.tracking_executable,
                        "-m", options.model_filename,
                        "-w", options.weight_filename,
                        "-o", options.result_filename])
        else:
            try:
                import commentjson as json
            except ImportError:
                import json
            import dpct
            import hytra.core.jsongraph

            with open(options.model_filename, 'r') as f:
                model = json.load(f)

            with open(options.weight_filename, 'r') as f:
                weights = json.load(f)

            result = dpct.trackFlowBased(model, weights)
            hytra.core.jsongraph.writeToFormattedJSON(options.result_filename, result)


    extra_params = []
    if options.do_merger_resolving:
        logging.info("Run merger resolving")
        check_call(["python", os.path.abspath("run_merger_resolving.py"), "--config", options.config_file])

        for p in ["--out-graph-json-file", "--out-label-image-file", "--out-result-json-file"]:
            index = unknown.index(p)
            extra_params.append(p.replace('--out-', '--'))
            extra_params.append(unknown[index + 1])

    if options.export_format is not None:
        logging.info("Convert result to {}...".format(options.export_format))
        if options.export_format in ['ilastikH5', 'ctc']:
            check_call(["python", os.path.abspath("json_result_to_events.py"), "--config", options.config_file] + extra_params)
            if options.export_format == 'ctc':
                # check_call(["python", os.path.abspath("ctc/hdf5_to_ctc.py"), "--config", options.config_file] + extra_params)
                check_call(["python", os.path.abspath("ctc/json_result_to_ctc.py"), "--config", options.config_file] + extra_params)
        elif options.export_format == 'labelimage':
            check_call(["python", os.path.abspath("json_result_to_labelimage.py"), "--config", options.config_file] + extra_params)
        elif options.export_format is not None:
            logging.error("Unknown export format chosen!")
            raise ValueError("Unknown export format chosen!")

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
    parser.add_argument("--tracking-executable", dest='tracking_executable', default=None,
                        type=str, help='executable that can run tracking based on JSON specified models')
    parser.add_argument('--graph-json-file', type=str, dest='model_filename',
                        help='Filename of the json graph description')
    parser.add_argument('--result-json-file', type=str, dest='result_filename',
                        help='Filename of the json file containing results')
    parser.add_argument('--weight-json-file', type=str, dest='weight_filename',
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
