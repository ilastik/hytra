import numpy as np
import logging
import configargparse as argparse
import glob
from skimage.external import tifffile
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
import copy

def edges_visualisation(options):
    '''
    Script for visualising the WRONG EDGES signalled by the CTC evaluation tool in TRA_log.txt.
    For this we show raw data, segmentation, the segmentation after tracking and the groundtruth for frame t and t+1.

    Note! Everything has to be in CTC-Format!!

    The edges to be added will be shown in Cyan in the groundtruth.
    The redundant edges to be deleted in tomato orange in the tracking result.
    The wrong semantics will be shown in gold in the tracking result.

    The output images for all problematic frames will be written in the ouptut directory. If this directory
    is not specified, then the results open in normal pyplot pop-up windows.
    '''

    if options.output == None:
        logging.getLogger('wrong_edges_visualisation.py').info("Figures will open in pop-up windows. Specify an output path if you want to save them all.")

    with open(options.txt_path) as txt:
        content = txt.readlines()

    edges_to_be_added = []
    redundant_edges = []
    incorrect_semantics = []
    problematic_frames = []
    set_redundant_edges = False
    set_edges_to_be_added = False
    set_edges_incorrect_semantics = False
    for line in content:
        if 'Edges To Be Added' in line:
            set_edges_to_be_added = True
            continue
        elif 'Redundant Edges To Be Deleted' in line:
            set_edges_to_be_added = False
            set_redundant_edges = True
            continue
        elif 'Edges with Incorrect Semantics' in line:
            set_redundant_edges = False
            set_edges_to_be_added = False
            set_edges_incorrect_semantics = True
            continue
        elif '====' in line:
            break # reached the end of the error part or TRA_log.txt
            
        if set_redundant_edges or set_edges_to_be_added or set_edges_incorrect_semantics:
            fromFrame, toFrame = line.split(' -> ')
            # reading both from- and toFrame is sensible, because they are not always consecutive
            t = '*'+fromFrame[3:5].strip().zfill(3)
            t_next = '*'+toFrame[3:5].strip().zfill(3)
            if [t, t_next] not in problematic_frames:
                problematic_frames.append([t, t_next])

        if set_edges_to_be_added:
            if [t, t_next] not in edges_to_be_added:
                edges_to_be_added.append([t, t_next])
            index = edges_to_be_added.index([t, t_next])
            # create a list that contains the frames pattern as first two elements and the GT labels of the last two
            edges_to_be_added[index].append(int(fromFrame[-3:-1].strip('=')))
            edges_to_be_added[index].append(int(toFrame[-4:-2].strip('=')))
        elif set_redundant_edges:
            if [t, t_next] not in redundant_edges:
                redundant_edges.append([t, t_next])
            index = redundant_edges.index([t, t_next])
            # create a list that contains the frames pattern as first two elements and the GT labels of the last two
            redundant_edges[index].append(int(fromFrame[-3:-1].strip('=')))
            redundant_edges[index].append(int(toFrame[-4:-2].strip('=')))
        elif set_edges_incorrect_semantics:
            if [t, t_next] not in incorrect_semantics:
                incorrect_semantics.append([t, t_next])
            index = incorrect_semantics.index([t, t_next])
            # create a list that contains the frames pattern as first two elements and the GT labels of the last two
            incorrect_semantics[index].append(int(fromFrame[-3:-1].strip('=')))
            incorrect_semantics[index].append(int(toFrame[-4:-2].strip('=')))

    for frames in sorted(problematic_frames):
        # extract the frame number
        logging.getLogger('wrong_edges_visualisation.py').debug("Looking at error {}".format(frames))
        rawim = []
        gtim = []
        segim =[]
        traim = []
        for i in range(2):
            rawim.append(tifffile.imread(glob.glob(options.raw_path + frames[i] + '.tif')))
            gtim.append(tifffile.imread(glob.glob(options.gt_path +frames[i] + '.tif')))
            segim.append(tifffile.imread(glob.glob(options.seg_path +frames[i] +'.tif')))
            traim.append(tifffile.imread(glob.glob(options.tra_path +frames[i] +'.tif')))

        # generating a suitable colormap for showing errors of the tracking result
        colors0 = np.array([[0,0,0]])  # background
        colors1 = np.array([[0,0,0]]) 
        for grayval in range(1,traim[0].max()+1):
            colors0 = np.vstack((colors0, [0,float(grayval)/traim[0].max(), 1-float(grayval)/traim[0].max()]))
        for grayval in range(1,traim[1].max()+1):
            colors1 = np.vstack((colors1, [0,float(grayval)/traim[1].max(), 1-float(grayval)/traim[1].max()]))

        tra_colors0 = colors0.copy()
        tra_colors1 = colors1.copy()
        # visualise edges to be added as cyan blobs in the tracking result
        if any(frames == [x[0],x[1]] for x in edges_to_be_added):
            logging.getLogger('wrong_edges_visualisation.py').debug("EDGE TO BE ADDED") 
            errors = [element for element in edges_to_be_added if [element[0], element[1]] == frames]
            for error in errors:
                colors0[(int(error[2]))] = [0.5, 1, 1] # error[2] is the label
                colors1[(int(error[3]))] = [0.5, 1, 1] # error[2] is the label
        # edges to be deleted as tomato orange blobs in the tracking result
        if any(frames == [x[0],x[1]] for x in redundant_edges):
            logging.getLogger('wrong_edges_visualisation.py').debug("REDUNDANT EDGE") 
            errors = [element for element in redundant_edges if [element[0], element[1]] == frames]
            for error in errors:
                tra_colors0[(int(error[2]))] = [1, 0.2, 0.1]
                tra_colors1[(int(error[3]))] = [1, 0.2, 0.1]
        # edges with incorrect semantics in gold in the tracking result
        if any(frames == [x[0],x[1]] for x in incorrect_semantics):
            logging.getLogger('wrong_edges_visualisation.py').debug("INCORRECT SEMANTICS") 
            errors = [element for element in incorrect_semantics if [element[0], element[1]] == frames]
            for error in errors:
                tra_colors0[(int(error[2]))] = [1, 0.95, 0]
                tra_colors1[(int(error[3]))] = [1, 0.95, 0]

        colormap0 = matplotlib.colors.ListedColormap(colors0, N=traim[0].max()+1)
        colormap1 = matplotlib.colors.ListedColormap(colors1, N=traim[1].max()+1)
        tracolormap0 = matplotlib.colors.ListedColormap(tra_colors0, N=traim[0].max()+1)
        tracolormap1 = matplotlib.colors.ListedColormap(tra_colors1, N=traim[1].max()+1)


        fig, axes = plt.subplots(nrows=2, ncols=4)

        # cmap can/should be adjusted here. 'flag' is best for Fluo-SIM 01
        name, raw0, axraw0= tifffile.imshow(rawim[0], figure=fig, subplot=241, title='raw image FRAME {}'.format(frames[0].strip('*')), cmap='flag')
        axraw0.colorbar.remove()
        raw0.axis('off')

        name, seg0, axseg0 = tifffile.imshow(segim[0], figure=fig, subplot=242, title='segmentation', cmap=colormap0, vmin=0, vmax=gtim[0].max()+1)
        axseg0.colorbar.remove()
        seg0.axis('off')

        name, tra0, axtra0 = tifffile.imshow(traim[0], figure=fig, subplot=243, title='tracking result', cmap=tracolormap0, vmin=0, vmax=traim[0].max()+1)
        axtra0.colorbar.remove()
        tra0.axis('off')

        name, gt0, axgt0 = tifffile.imshow(gtim[0], figure=fig, subplot=244, title='groundtruth', cmap=colormap0, vmin=0, vmax=gtim[1].max()+1)
        axgt0.colorbar.remove()
        gt0.axis('off')

        name, raw1, axraw1= tifffile.imshow(rawim[1], figure=fig, subplot=245, title='raw image FRAME {}'.format(frames[1].strip('*')), cmap='flag')
        axraw1.colorbar.remove()
        raw1.axis('off')

        name, seg1, axseg1 = tifffile.imshow(segim[1], figure=fig, subplot=246, title='segmentation', cmap=colormap1, vmin=0, vmax=gtim[1].max()+1)
        axseg1.colorbar.remove()
        seg1.axis('off')

        name, tra1, axtra1 = tifffile.imshow(traim[1], figure=fig, subplot=247, title='tracking result', cmap=tracolormap1, vmin=0, vmax=traim[1].max()+1)
        axtra1.colorbar.remove()
        tra1.axis('off')

        name, gt1, axgt1 = tifffile.imshow(gtim[1], figure=fig, subplot=248, title='groundtruth', cmap=colormap1, vmin=0, vmax=gtim[1].max()+1)
        axgt1.colorbar.remove()
        gt1.axis('off')

        fig.tight_layout()
        plt.subplots_adjust(wspace = 0.2)

        if options.output == None:
            plt.show()
        else:
            plt.savefig(options.output+'/error_frame{}_to_{}'.format(error[0].strip('*'), error[1].strip('*')))
            plt.close()
    
    if options.output != None:
        logging.getLogger('wrong_edges_visualisation.py').info("Done writing the images. You can open them to view the errors.")

if __name__ == '__main__':
    """
    Visualise the erronate edges by viewing raw data, groundtruth and segmentation.
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

    edges_visualisation(options)
