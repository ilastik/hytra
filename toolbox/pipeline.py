import configargparse as argparse
import logging
from subprocess import check_call

def run_pipeline(options):
    if options.do_ctc_groundtruth_conversion:
        logging.info("Convert CTC groundtruth to our format...")
        check_call(["python", "ctc_gt_to_hdf5.py", "--config", options.config_file])
    
    if options.do_ctc_raw_data_conversion:
        logging.info("Convert CTC raw data to HDF5...")
        check_call(["python", "stack_to_h5.py", "--config", options.config_file])
    
    if options.do_ctc_segmentation_conversion:
        logging.info("Convert CTC segmentation to HDF5...")
        check_call(["python", "segmentation_to_hdf5.py", "--config", options.config_file])
    
    if options.do_train_transition_classifier:
        logging.info("Train transition classifier...")
        check_call(["python", "train_transition_classifier.py", "--config", options.config_file])
    
    if options.do_create_graph:
        logging.info("Create hypotheses graph...")
        check_call(["python", "hypotheses_graph_to_json.py", "--config", options.config_file])

    if options.do_tracking:
        logging.info("Run tracking...")
        check_call([options.tracking_executable, 
                    "-m", options.model_filename, 
                    "-w", options.weight_filename,
                    "-o", options.result_filename])

    if options.export_format is not None:
        logging.info("Convert result to {}...".format(options.export_format))
        if options.export_format in ['ilastikH5', 'ctc']:
            check_call(["python", "json_result_to_events.py", "--config", options.config_file])
            if options.export_format == 'ctc':
                check_call(["python", "hdf5_to_ctc.py", "--config", options.config_file])
        elif options.export_format == 'labelimage':
            check_call(["python", "json_result_to_labelimage.py", "--config", options.config_file])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Cell Tracking Challenge raw data to a HDF5 stack',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file', required=True)
    
    parser.add_argument("--do-ctc-groundtruth-conversion", dest='do_ctc_groundtruth_conversion', action='store_true', default=False)
    parser.add_argument("--do-ctc-raw-data-conversion", dest='do_ctc_raw_data_conversion', action='store_true', default=False)
    parser.add_argument("--do-ctc-segmentation-conversion", dest='do_ctc_segmentation_conversion', action='store_true', default=False)
    parser.add_argument("--do-train-transition-classifier", dest='do_train_transition_classifier', action='store_true', default=False)
    parser.add_argument("--do-create-graph", dest='do_create_graph', action='store_true', default=False)
    parser.add_argument("--do-tracking", dest='do_tracking', action='store_true', default=False)
    parser.add_argument("--export-format", dest='export_format', type=str, default=None,
                        help='Export format may be one of: "ilastikH5", "ctc", "labelimage", or None')
    parser.add_argument("--tracking-executable", dest='tracking_executable', required=True,
                        type=str, help='executable that can run tracking based on JSON specified models')
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

    run_pipeline(options)