import logging
from libdvid import DVIDNodeService, DVIDServerService
import h5py
import numpy as np
import json_tricks.np as json
from pluginsystem.plugin_manager import TrackingPluginManager

if __name__ == "__main__":
    """
    download raw data and segmentation of a dataset from dvid.

    Example: python dvid/download_dataset.py --dvid-address 104.196.46.138:80 --label-image seg.ilp --raw raw.h5 --raw-path data --uuid 2994598cb92e446caa8d40a32c76b060 --time-range 0 10
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download raw data and segmentation from dvid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uuid",
        required=True,
        type=str,
        dest="uuid",
        help="Datset UUID that can be found on the DVID web interface",
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
        help="Filename of the HDF5 file that will contain the label images",
    )
    parser.add_argument(
        "--raw",
        required=True,
        type=str,
        dest="rawFilename",
        help="Filename of the hdf5 file that will contain the raw data",
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
        "--time-range",
        type=int,
        nargs=2,
        dest="timeRange",
        help="Set time range to download (inclusive!)",
    )
    parser.add_argument(
        "--verbose", type=bool, dest="verbose", default=False, help="verbose logs"
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # get node service
    server_address = args.dvidAddress
    node_service = DVIDNodeService(server_address, args.uuid)

    keyvalue_store = "config"
    settings = json.loads(node_service.get(keyvalue_store, "imageInfo"))

    shape = settings["shape"]
    time_range = settings["time_range"]
    if args.timeRange is not None:
        time_range = (
            max(time_range[0], args.timeRange[0]),
            min(time_range[1], args.timeRange[1]),
        )

    logging.info(
        "Downloading time range {} to {} of shape {}".format(
            time_range, server_address, shape
        )
    )

    raw_data = np.zeros((time_range[1] - time_range[0], shape[0], shape[1], shape[2]))

    # download all frames
    with h5py.File(args.ilpFilename, "w") as seg_h5:
        for frame in range(time_range[0], time_range[1]):
            logging.info("Downloading frame {}".format(frame))

            raw_name = "raw-{}".format(frame)
            seg_name = "seg-{}".format(frame)
            raw_image = node_service.get_gray3D(raw_name, shape, (0, 0, 0))
            seg_image = node_service.get_labels3D(seg_name, shape, (0, 0, 0))

            group_name = args.labelImagePath % (
                frame,
                frame + 1,
                shape[0],
                shape[1],
                shape[2],
            )
            seg_h5.create_dataset(group_name, data=seg_image, dtype=np.uint32)
            raw_data[frame, ...] = raw_image

    with h5py.File(args.rawFilename, "w") as raw_h5:
        raw_h5.create_dataset(args.rawPath, data=raw_data, dtype=np.uint8)
