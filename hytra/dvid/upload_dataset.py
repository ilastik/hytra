import logging
from libdvid import DVIDNodeService, DVIDServerService
import numpy as np
import json_tricks.np as json
from pluginsystem.plugin_manager import TrackingPluginManager


def dataToBlock(data, dtype=np.uint8, block_size=32):
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=2)
    elif len(data.shape) != 3:
        raise ValueError("Cannot prepare data of shape that is not a 2D or 3D frame")

    old_shape = data.shape
    new_shape = []
    for d in old_shape:
        if d % block_size == 0:
            new_shape.append(d)
        else:
            new_shape.append(((d // block_size) + 1) * block_size)

    logging.debug("transformed data from {} to {}".format(old_shape, new_shape))
    new_data = np.zeros(new_shape, dtype=dtype)
    new_data[0 : old_shape[0], 0 : old_shape[1], 0 : old_shape[2]] = data
    return new_data


if __name__ == "__main__":
    """
    Upload raw data and segmentation of a dataset to dvid.

    Example: python dvid/upload_dataset.py --dvid-address 104.196.46.138:80 --label-image /Users/chaubold/hci/data/animal-tracking/FlyBowlTracking/FlyBowlTracking.ilp --raw /Users/chaubold/hci/data/animal-tracking/FlyBowlTracking/FlyBowlMovie.h5 --raw-path data --dataset-name flybowl-test-2016-04-07 --time-range 0 10
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload raw data and segmentation to dvid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        type=str,
        dest="datasetName",
        help="Datset name that will be seen in the DVID web interface",
    )
    parser.add_argument(
        "--dvid-address",
        required=True,
        type=str,
        dest="dvidAddress",
        help="<IP>:<Port> of the dvid server",
    )
    parser.add_argument(
        "--label-image",
        required=True,
        type=str,
        dest="ilpFilename",
        help="Filename of the HDF5 file containing the label images",
    )
    parser.add_argument(
        "--raw",
        required=True,
        type=str,
        dest="rawFilename",
        help="Filename of the hdf5 file containing the raw data",
    )
    parser.add_argument(
        "--raw-path",
        required=True,
        type=str,
        dest="rawPath",
        help="Path inside HDF5 file to raw volume",
    )
    parser.add_argument(
        "--label-image-path",
        type=str,
        dest="labelImagePath",
        help="Path inside ilastik project file to the label image",
        default="/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]",
    )
    parser.add_argument(
        "--object-count-classifier-file",
        type=str,
        dest="objectCountClassifierFile",
        help="HDF5 file containing the object count classifier",
        default=None,
    )
    parser.add_argument(
        "--object-count-classifier-path",
        type=str,
        dest="objectCountClassifierPath",
        help="Path inside HDF5 file to the object count classifier",
        default="/CountClassification",
    )
    parser.add_argument(
        "--rf-zero-padding",
        type=int,
        dest="rfZeroPadding",
        default=4,
        help="Number of digits per forest index inside the ClassifierForests HDF5 group",
    )
    parser.add_argument(
        "--time-range",
        type=int,
        nargs=2,
        dest="timeRange",
        help="Set time range to upload (inclusive!)",
    )
    parser.add_argument(
        "--verbose", type=bool, dest="verbose", default=False, help="verbose logs"
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialize plugin manager
    plugin_manager = TrackingPluginManager(verbose=False)
    image_provider = plugin_manager.getImageProvider()

    # create dataset on server and get uuid
    server_address = args.dvidAddress
    server_service = DVIDServerService(server_address)
    uuid = server_service.create_new_repo(args.datasetName, "description")
    logging.info("UUID:\n{}".format(uuid))

    # get node service
    node_service = DVIDNodeService(server_address, uuid)

    # get dataset size and store in dvid
    shape = image_provider.getImageShape(args.ilpFilename, args.labelImagePath)
    time_range = image_provider.getTimeRange(args.ilpFilename, args.labelImagePath)
    if args.timeRange is not None:
        time_range = (
            max(time_range[0], args.timeRange[0]),
            min(time_range[1], args.timeRange[1]),
        )
    logging.info("Uploading time range {} to {}".format(time_range, server_address))
    keyvalue_store = "config"
    node_service.create_keyvalue(keyvalue_store)
    settings = {"shape": shape, "time_range": time_range}
    node_service.put(keyvalue_store, "imageInfo", json.dumps(settings))

    # upload all frames
    for frame in range(time_range[0], time_range[1]):
        logging.info("Uploading frame {}".format(frame))
        label_image = image_provider.getLabelImageForFrame(
            args.ilpFilename, args.labelImagePath, frame
        )
        raw_image = image_provider.getImageDataAtTimeFrame(
            args.rawFilename, args.rawPath, frame
        )

        raw_name = "raw-{}".format(frame)
        seg_name = "seg-{}".format(frame)
        node_service.create_grayscale8(raw_name)
        node_service.put_gray3D(
            raw_name, dataToBlock(raw_image, dtype=np.uint8), (0, 0, 0)
        )
        node_service.create_labelblk(seg_name)
        node_service.put_labels3D(
            seg_name, dataToBlock(label_image, dtype=np.uint64), (0, 0, 0)
        )

    # TODO: upload classifier
