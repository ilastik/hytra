from __future__ import unicode_literals
import numpy as np
import logging
import configargparse as argparse
import glob
from skimage.external import tifffile
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
import copy

def error_visualisation(options):
    '''
    Script for visualising the false negatives signalled by the CTC evaluation tool in TRA_log.txt.
    For this we show raw data, segmentation, the segmentation after tracking and the groundtruth.

    Note! Everything has to be in CTC-Format!!

    The false negatives (from tracking) will be shown in red in the groundtruth and in the segmentation.
    The false positives from the segmentation will appear pink in the segmentation.
    The mergers are shown in purple on the tracking image.

    The output images for all problematic frames will be written in the ouptut directory. If this directory
    is not specified, then the results open in normal pyplot pop-up windows.
    '''

    if options.output == None:
        logging.getLogger('error_visualisation.py').info("Figures will open in pop-up windows. Specify an output path if you want to save them all.")

    with open(options.txt_path) as txt:
        content = txt.readlines()

    problematic = []
    merger = {}
    for line in content:
        if "[T=" in line:
            break
        if "T=" in line:
            if ' Label=' in line:
                merger.setdefault('*'+line[2:4].strip().zfill(3), [])
                merger['*'+line[2:4].strip().zfill(3)].append(line[-3:-1].strip('='))
            problematic.append('*'+line[2:4].strip().zfill(3))

    for frame in sorted(list(set(problematic))):
        # extract the frame number
        logging.getLogger('error_visualisation.py').debug("Looking at frame {}".format(frame))

        rawim = tifffile.imread(glob.glob(options.raw_path+frame+'.tif'))
        gtim = tifffile.imread(glob.glob(options.gt_path+frame+'.tif'))
        segim = tifffile.imread(glob.glob(options.seg_path+frame+'.tif'))
        traim = tifffile.imread(glob.glob(options.tra_path+frame+'.tif'))

        # generating a suitable colormap for showing errors of the tracking result
        colors = np.array([[0,0,0]])  # background
        for grayval in range(1,traim.max()+1):
            colors = np.vstack((colors, [0,float(grayval)/traim.max(), 1-float(grayval)/traim.max()]))

        tra_colors = colors.copy()
        # visualise mergers as purple blobs in the tracking result
        if frame in merger.keys():
            for label in merger[frame]:
                tra_colors[(int(label))] = [0.65, 0, 1]
        tracolormap = matplotlib.colors.ListedColormap(tra_colors, N=traim.max()+1)
        
        # groundtruth colormap
        err_colors = np.array([[0,0,0]])
        for grayval in range(1, gtim.max()+1):
            position = np.where(gtim == grayval)
            if position[0].shape[0]==0:
                err_colors = np.vstack((err_colors, [1,0,0]))
            else:
                act_grayval = int(ceil(np.median(traim[position])))
                if act_grayval == 0:
                    err_colors = np.vstack((err_colors, [1,0,0])) # false negatives in tracking result
                else:
                    err_colors = np.vstack((err_colors, colors[act_grayval]))
        gtcolormap = matplotlib.colors.ListedColormap(err_colors, N=gtim.max()+1)

        # creating colormap fot the segmentation
        seg_colors = np.array([[0,0,0]])
        for grayval in range(1, segim.max()+1):
            position = np.where(segim == grayval)
            if position[0].shape[0]==0:
                seg_colors = np.vstack((seg_colors, [1,0.8,0.9]))
            else:
                act_grayval = int(ceil(np.median(gtim[position])))
                if act_grayval == 0:
                    seg_colors = np.vstack((seg_colors, [1,0.8,0.9])) # false positives will appear pink
                else:
                    seg_colors = np.vstack((seg_colors, err_colors[act_grayval]))
        segcolormap = matplotlib.colors.ListedColormap(seg_colors, N=gtim.max()+1)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        # cmap can/should be adjusted here. 'flag' is best for Fluo-SIM 01
        name, raw, axraw= tifffile.imshow(rawim, figure=fig, subplot=221, title='raw image FRAME {}'.format(frame.strip('*')), cmap='flag')
        axraw.colorbar.remove()
        raw.axis('off')

        name, seg, axseg = tifffile.imshow(segim, figure=fig, subplot=222, title='segmentation', cmap=segcolormap, vmin=0, vmax=gtim.max()+1)
        axseg.colorbar.remove()
        seg.axis('off')

        name, tra, axtra = tifffile.imshow(traim, figure=fig, subplot=223, title='tracking result', cmap=tracolormap, vmin=0, vmax=traim.max()+1)
        axtra.colorbar.remove()
        tra.axis('off')

        name, gt, axgt = tifffile.imshow(gtim, figure=fig, subplot=224, title='groundtruth', cmap=gtcolormap, vmin=0, vmax=gtim.max()+1)
        axgt.colorbar.remove()
        gt.axis('off')

        fig.tight_layout()
        plt.subplots_adjust(hspace = 0.2)

        if options.output == None:
            plt.show()
        else:
            plt.savefig(options.output+'/error_frame{}'.format(frame.strip('*')))
            plt.close()
    
    if options.output != None:
        logging.getLogger('error_visualisation.py').info("Done writing the images. You can open them to view the errors.")

if __name__ == '__main__':
    """
    Visualise the erros by viewing raw data, groundtruth and segmentation.
    """

    parser = argparse.ArgumentParser(
        description='Visualise the erros by viewing raw data, groundtruth segmentation and tracking result.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--res-txt-file', required=True, type=str, dest='txt_path',
                        help='filname of the ctc txt result file')
    parser.add_argument('--raw-path', type=str, dest='raw_path',
                        help='path of the raw image')
    parser.add_argument('--gt-path', required=True, type=str, dest='gt_path',
                        help='path to the groundtruth')
    parser.add_argument('--seg-path', required=True, type=str, dest='seg_path',
                        help='path to the segmentation')
    parser.add_argument('--tra-path', required=True, type=str, dest='tra_path',
                        help='path to the segmentation')
    parser.add_argument('--output', type=str, dest='output',
                        help='path of the output')

    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    error_visualisation(options)
