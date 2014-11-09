import optparse
import numpy as np
import h5py
import os
import scipy.misc

def generate_groundtruth(options):
    # read image

    # read ground truth file
    with h5py.File(options.input_file, 'r') as inputfile:

        if(options.end == -1):
            options.end = int(inputfile["tracking"].keys()[-1])
            print options.end


        inId_to_outId_dics = {}
        inId_to_outId_funct = {}

        splitdict = {}
        movedict = {}


        for t in xrange(options.start,options.end+1,1):

            outputFileName = options.output_dir.rstrip('/')+ "/%04d.h5" % t
            #create output file
            if os.path.exists(outputFileName):
                os.remove(outputFileName)

            with h5py.File(outputFileName, 'w') as out_h5:

                trackingdata = out_h5.create_group('tracking')
                meta = out_h5.create_group('objects/meta')

                with h5py.File(options.label_image, 'r') as labelfile:

                    outLabelImage = np.array(labelfile["exported_data_T"][t]).squeeze().astype(np.int)
                    inputLabelImage   = np.array(inputfile["label_image"][t]).squeeze().astype(np.int)

                    inputIds  =  np.unique(inputLabelImage).flatten()
                    outputIds =  np.unique(outLabelImage).flatten()

                    #remove background
                    inputIds = np.delete(inputIds,np.where(inputIds == 0))
                    outputIds = np.delete(outputIds,np.where(outputIds == 0))

                    meta.create_dataset("id", data=outputIds, dtype='u2')
                    meta.create_dataset("valid", data=np.ones(outputIds.shape[0]), dtype='u2')

                    #create id translation dictionary 

                    inId_to_outId_dics[t] = {}

                    applist = []
                    dislist = []
                    movlist = []
                    spllist = []
                    merlist = []

                    # for outId in outputIds:
                    #     if( outId > 0 ):
                    #         outId_to_inId_dics[t][outId] = np.unique(inputLabelImage[outLabelImage==outId]).tolist()
                    #         if(0 in outId_to_inId_dics[t][outId]):
                    #             outId_to_inId_dics[t][outId].remove(0)

                    #         if(len(outId_to_inId_dics[t][outId]) == 0):
                    #             del outId_to_inId_dics[t][outId]
                    #         elif(len(outId_to_inId_dics[t][outId]) == 1):
                    #             outId_to_inId_dics[t][outId] = outId_to_inId_dics[t][outId][0]
                    #         else:
                    #             merlist.append(outId_to_inId_dics[t][outId])
                    #             del outId_to_inId_dics[t][outId]

                    for inId in inputIds:
                        if( inId > 0 ):
                            inId_to_outId_dics[t][inId] = np.unique(outLabelImage[inputLabelImage==inId]).tolist()
                            if(0 in inId_to_outId_dics[t][inId]):
                                inId_to_outId_dics[t][inId].remove(0)

                            if(len(inId_to_outId_dics[t][inId]) == 0):
                                inId_to_outId_dics[t][inId] = -1
                            elif(len(inId_to_outId_dics[t][inId]) == 1):
                                inId_to_outId_dics[t][inId] = inId_to_outId_dics[t][inId][0]
                            else:
                                merlist.append(inId_to_outId_dics[t][inId])
                                inId_to_outId_dics[t][inId] = -2


                    print "merger", merlist
                    print inId_to_outId_dics[t]
                    inId_to_outId_funct[t] = np.vectorize(inId_to_outId_dics[t].get)

                    if(len(merlist) > 0):
                        #TODO: THIS IS WRONG NEED TO ADD NUMBER OF MERGERS AND OUT ID
                        trackingdata.create_dataset("Mergers", data=np.asarray(merlist), dtype='u2') 

                    if(len(inId_to_outId_funct) > 1):
                        if("Moves" in inputfile["tracking"]["{0:04d}".format(t)].keys()):
                            moves = np.array(inputfile["tracking"]["{0:04d}".format(t)]["Moves"]).squeeze().astype(np.int)
                            moves = np.reshape(moves, (-1,2))
                            moves[:,0] = inId_to_outId_funct[t-1](moves[:,0])
                            moves[:,1] = inId_to_outId_funct[t](moves[:,1])
                            trackingdata.create_dataset("Moves", data=moves, dtype='u2')
                            movedict[t] = moves


                        if("Splits" in inputfile["tracking"]["{0:04d}".format(t)].keys()):
                            splits = np.array(inputfile["tracking"]["{0:04d}".format(t)]["Splits"]).squeeze().astype(np.int)
                            splits = np.reshape(splits,(-1,3))
                            splits[:,0] = inId_to_outId_funct[t-1](splits[:,0])
                            splits[:,1:] = inId_to_outId_funct[t](splits[:,1:])
                            trackingdata.create_dataset("Splits", data=splits, dtype='u2') 
                            splitdict[t] = splits

                        #app = ids_t - move_{t-1}(0) - div_{t-1}(1) - div_{t-1}(2) 
                        applist    = [inId_to_outId_dics[t][i] for i in inputIds
                                        if ((t-1 in movedict and not i in movedict[t-1][:,1])
                                            or (t-1 in splitdict and  i in splitdict[t-1][:,1:]))]
                        disapplist = [inId_to_outId_dics[t][i] for i in inputIds
                                        if ((t   in movedict and not i in movedict[t  ][:,0])
                                            or (t   in splitdict and  i in splitdict[t  ][:,0 ]))]

                        if(len(applist) > 0):
                            trackingdata.create_dataset("Appearances", data=np.asarray(applist), dtype='u2')
                        if(len(disapplist) > 0):
                            trackingdata.create_dataset("Disappearances", data=np.asarray(disapplist), dtype='u2') 





# raw_image = np.array(raw_h5['/'.join(options.raw_path.split('/'))][timestep, ..., 0]).squeeze().astype(np.float32)

def unusedCode():

    if not options.output_file:
        options.output_file = options.input_dir + '/.h5'


    if os.path.exists(options.output_file):
        os.remove(options.output_file)

    f = h5py.File(options.output_file, 'r')

    with h5py.File(options.output_file, 'w') as out_h5:

        ids = out_h5.create_group('ids')
        tracking = out_h5.create_group('tracking')



        # object ids per frame
        objects_per_frame = []
        for frame in range(label_volume.shape[2]):
            objects = np.unique(label_volume[..., frame, 0])
            ids.create_dataset(format(frame, "04"), data=objects, dtype='u2')
            objects_per_frame.append(set(objects))

        # move, mitosis and split events
        tracking_frame = tracking.create_group(format(0, "04"))
        for frame in range(1, label_volume.shape[2]):
            tracking_frame = tracking.create_group(format(frame, "04"))

            # intersect track id sets of both frames, and place moves in HDF5 file
            tracks_in_both_frames = objects_per_frame[frame - 1] & objects_per_frame[frame] - set([0])
            moves = np.array([list(tracks_in_both_frames), list(tracks_in_both_frames)]).transpose()
            tracking_frame.create_dataset("Moves", data=moves, dtype='u2')

            # add the found splits as both, mitosis and split events
            if frame in split_events.keys():
                splits_in_frame = split_events[frame]
                mitosis = splits_in_frame.keys()
                if len(mitosis) > 0:
                    tracking_frame.create_dataset("Mitosis", data=np.array(mitosis), dtype='u2')
                splits = [[key] + value for key, value in splits_in_frame.iteritems()]
		# make sure all splits have the same dimension
		max_split_length = max(map(len, splits))
		min_split_length = min(map(len, splits))
		if min_split_length != max_split_length:
			print("In timestep {}: Found splits longer than minimum {}, cutting off children to make number equal!".format(frame, min_split_length))
			for i, split in enumerate(splits):
				splits[i] = split[0:min_split_length]


                if len(splits) > 0:
                    tracking_frame.create_dataset("Splits", data=np.array(splits), dtype='u2')



if __name__ == "__main__":
    parser = optparse.OptionParser(description='Compute TRA loss of a new labeling compared to ground truth')

    # file paths
    parser.add_option('--output-dir', type=str, dest='output_dir', default=".",
                        help='Folder where the groundTruthfiles are created')
    parser.add_option('--input-file', type=str, dest='input_file',
                        help='Filename for the resulting HDF5 file.')

    parser.add_option('--start', type=int, dest='start',
                        help='first timestep',default=0)
    parser.add_option('--end', type=int, dest='end',
                        help='last timestep',default=-1)
    parser.add_option('--label-image', type=str, dest='label_image',
                        help='file to label image with ids corresponding to opengm ids. ')

    # parse command line
    opt , args = parser.parse_args()

    generate_groundtruth(opt)
