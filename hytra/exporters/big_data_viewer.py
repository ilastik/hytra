import xml.etree.ElementTree as ET
import argparse

if __name__ == '__main__':
	"""
	Transform tracking result and ilastik raw + segmentation data to the BigDataViewer format
    """

    parser = argparse.ArgumentParser(description='Transform tracking result and ilastik raw'
    								 '+ segmentation data to the BigDataViewer format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ilastik-project', required=True, type=str, dest='ilpFilename',
                        help='Filename of the ilastik project')
    parser.add_argument('--raw', required=True, type=str, dest='rawFilename',
                        help='Filename of the hdf5 file containing the raw data')
    parser.add_argument('--raw-path', required=True, type=str, dest='rawPath',
                        help='Path inside HDF5 file to raw volume')
    parser.add_argument('--label-image-path', type=str, dest='labelImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument('--object-count-classifier-path', type=str, dest='objectCountClassifierPath',
                        help='Path inside ilastik project file to the object count classifier',
                        default='/CountClassification')
    parser.add_argument('--division-classifier-path', type=str, dest='divisionClassifierPath',
                        help='Path inside ilastik project file to the division classifier',
                        default='/DivisionDetection')
    parser.add_argument('--without-divisions', dest='withoutDivisions', action='store_true',
                        help='Specify this if no divisions are allowed in this dataset',
                        default=False)
    parser.add_argument('--rf-zero-padding', type=int, dest='rfZeroPadding', default=4,
                        help='Number of digits per forest index inside the ClassifierForests HDF5 group')

    parser.add_argument('--image-provider', type=str, dest='image_provider_name', default="LocalImageLoader")
    parser.add_argument('--feature-serializer', type=str, dest='feature_serializer_name', default='LocalFeatureSerializer')

    args = parser.parse_args()

    filename = args.rawPath

    root = ET.Element('SpimData')
    root.set('version', '0.2')

    # set up base path
    basePath = ET.SubElement(root, 'BasePath')
    basePath.set('type', 'relative')
    basePath.text = '.'

    # describe datasets
    sequenceDesc = ET.SubElement(root, 'SequenceDescription')
    imageLoader = ET.SubElement(sequenceDesc, 'ImageLoader')
    imageLoader.set('format', 'bdv.hdf5')
    hdfFile = ET.SubElement(imageLoader, 'hdf5')
    hdfFile.set('type', 'relative')
    hdfFile.text = filename

    # describe transformations (which are just the identity for us!)
    viewRegistrations = ET.SubElement(root, 'ViewRegistrations')