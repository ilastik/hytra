# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import commentjson as json
import logging
import configargparse as argparse
import numpy as np
import h5py
from multiprocessing import Pool
import hytra.core.jsongraph

def writeEvents(timestep, activeLinks, activeDivisions, mergers, detections, fn, labelImagePath, ilpFilename):
    dis = []
    app = []
    div = []
    mov = []
    mer = []
    mul = []
    
    logging.getLogger('json_result_to_events.py').debug("-- Writing results to {}".format(fn))
    try:
        # convert to ndarray for better indexing
        dis = np.asarray(dis)
        app = np.asarray(app)
        div = np.asarray([[k, v[0], v[1]] for k,v in activeDivisions.iteritems()])
        mov = np.asarray(activeLinks)
        mer = np.asarray([[k,v] for k,v in mergers.iteritems()])
        mul = np.asarray(mul)

        with h5py.File(ilpFilename, 'r') as src_file:
            # find shape of dataset
            shape = src_file['/'.join(labelImagePath.split('/')[:-1])].values()[0].shape[1:4]

            with h5py.File(fn, 'w') as dest_file:
                # write meta fields and copy segmentation from project
                li_name = labelImagePath % (timestep, timestep + 1, shape[0], shape[1], shape[2])
                label_img = np.array(src_file[li_name][0, ..., 0]).squeeze()
                seg = dest_file.create_group('segmentation')
                seg.create_dataset("labels", data=label_img)
                meta = dest_file.create_group('objects/meta')
                ids = np.unique(label_img)
                ids = ids[ids > 0]
                valid = np.ones(ids.shape)
                meta.create_dataset("id", data=ids, dtype=np.uint32)
                meta.create_dataset("valid", data=valid, dtype=np.uint32)

                tg = dest_file.create_group("tracking")

                # write associations
                if app is not None and len(app) > 0:
                    ds = tg.create_dataset("Appearances", data=app, dtype=np.int32)
                    ds.attrs["Format"] = "cell label appeared in current file"

                if dis is not None and len(dis) > 0:
                    ds = tg.create_dataset("Disappearances", data=dis, dtype=np.int32)
                    ds.attrs["Format"] = "cell label disappeared in current file"

                if mov is not None and len(mov) > 0:
                    ds = tg.create_dataset("Moves", data=mov, dtype=np.int32)
                    ds.attrs["Format"] = "from (previous file), to (current file)"

                if div is not None and len(div) > 0:
                    ds = tg.create_dataset("Splits", data=div, dtype=np.int32)
                    ds.attrs["Format"] = "ancestor (previous file), descendant (current file), descendant (current file)"

                if mer is not None and len(mer) > 0:
                    ds = tg.create_dataset("Mergers", data=mer, dtype=np.int32)
                    ds.attrs["Format"] = "descendant (current file), number of objects"

                if mul is not None and len(mul) > 0:
                    ds = tg.create_dataset("MultiFrameMoves", data=mul, dtype=np.int32)
                    ds.attrs["Format"] = "from (given by timestep), to (current file), timestep"

        logging.getLogger('json_result_to_events.py').debug("-> results successfully written")
    except Exception as e:
        logging.getLogger('json_result_to_events.py').warning("ERROR while writing events: {}".format(str(e)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take a json file containing a result to a set of HDF5 events files',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--result-json-file', required=True, type=str, dest='result_filename',
                        help='Filename of the json file containing results')
    parser.add_argument('--label-image-file', required=True, type=str, dest='ilp_filename',
                        help='Filename of the ilastik-style segmentation HDF5 file')
    parser.add_argument('--label-image-path', dest='label_img_path', type=str,
                        default='/ObjectExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]',
                        help='internal hdf5 path to label image')
    parser.add_argument('--h5-event-out-dir', type=str, dest='out_dir', default='.', help='Output directory for HDF5 files')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)
    
    args, unknown = parser.parse_known_args()

    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    with open(args.result_filename, 'r') as f:
        result = json.load(f)
        assert(result['detectionResults'] is not None)
        assert(result['linkingResults'] is not None)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('json_result_to_events.py').debug("Ignoring unknown parameters: {}".format(unknown))

    traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(model)
    # timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    # there might be empty frames. We want them as output too. A little messy, but:
    timesteps = [str(t).decode("utf-8") for t in range(int(min(traxelIdPerTimestepToUniqueIdMap.keys())) , int(max(traxelIdPerTimestepToUniqueIdMap.keys()))+1 )]

    mergers, detections, links, divisions = hytra.core.jsongraph.getMergersDetectionsLinksDivisions(result, uuidToTraxelMap)

    # group by timestep for event creation
    mergersPerTimestep = hytra.core.jsongraph.getMergersPerTimestep(mergers, timesteps)
    linksPerTimestep = hytra.core.jsongraph.getLinksPerTimestep(links, timesteps)
    detectionsPerTimestep = hytra.core.jsongraph.getDetectionsPerTimestep(detections, timesteps)
    divisionsPerTimestep = hytra.core.jsongraph.getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps)
    
    # save to disk in parallel
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    processing_pool = Pool()
    for timestep in timesteps:
        fn = os.path.join(args.out_dir, "{0:05d}.h5".format(int(timestep)))
        processing_pool.apply_async(writeEvents,
                                    (int(timestep),
                                     linksPerTimestep[timestep], 
                                     divisionsPerTimestep[timestep], 
                                     mergersPerTimestep[timestep], 
                                     detectionsPerTimestep[timestep], 
                                     fn, 
                                     args.label_img_path, 
                                     args.ilp_filename))

    processing_pool.close()
    processing_pool.join()

