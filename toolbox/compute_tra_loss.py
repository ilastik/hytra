#!/usr/bin/env python
import sys

sys.path.append('../.')
sys.path.append('.')

import argparse
import h5py
import numpy as np
import time
import pickle
import logging

def check_detections_frame(gt_labels, timestep, frame_labels, threshold):
    import numpy as np
    import time

    def compute_overlap_ratios(label_image_a, label_image_b, label_a):
        """
        Compute the ratio of overlapping pixels of label a in image a, and label b in image b,
        for all b's that are present in the overlap. Ratio = intersection / size(image_b[label_b])
        Return a list of tuples (label, ratio)
        """
        assert(label_image_a.shape == label_image_b.shape)

        # count how many pixels of label_image_b that were assigned to label_a in the other image,
        # are actually assigned to label_b
        tmp = label_image_b[label_image_a == label_a]
        label_set = set(tmp) - set([0])

        overlap_ratios = []

        for label_b in label_set:
            overlap_pixel_count = np.sum(tmp == label_b)
            label_b_count = np.sum(label_image_b == label_b)
            ratio = float(overlap_pixel_count) / float(label_b_count)
            overlap_ratios.append((label_b, ratio))

        # return ratio
        return overlap_ratios

    print("---------------------------------\nChecking Timestep: " + str(timestep) + '\n')
    start_time = time.time()

    # prepare outputs
    gt2frame_assignments = {}
    frame2gt_assignments = {}
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    needed_splits = 0

    # compare
    gt_labels_unmatched = {}
    for gt_label in set(np.unique(gt_labels)) - set([0]):
        gt_labels_unmatched[gt_label] = 1

    # find all overlap candidates and compute TRA measures from that
    for frame_label in set(np.unique(frame_labels)) - set([0]):
        candidates = compute_overlap_ratios(frame_labels.squeeze(), gt_labels.squeeze(), frame_label)
        # filter candidates by threshold:
        candidates = [(l, r) for l, r in candidates if r > threshold]

        frame2gt_assignments[frame_label] = []
        for l, r in candidates:
            gt2frame_assignments[l] = [frame_label]
            frame2gt_assignments[frame_label].append(l)
            gt_labels_unmatched[l] = 0
            true_positives += 1

        if len(candidates) == 0:
            print("False Positive label {} at timestep {}".format(frame_label, timestep))
            false_positives += 1
        elif len(candidates) > 1:
            print("Split needed of label {} at timestep {}".format(frame_label, timestep))
            needed_splits += len(candidates) - 1

    false_negatives = sum(gt_labels_unmatched.values())
    for i, val in gt_labels_unmatched.iteritems():
        if val == 0:
            continue
        print("False Negative gt label {} at timestep {}".format(i, timestep))

    end_time = time.time()
    print("Checking overlaps for frame {} took {} secs".format(timestep, end_time - start_time))
    return (false_negatives, false_positives, true_positives, needed_splits, gt2frame_assignments, frame2gt_assignments, timestep)


def check_detections(options):
    gt2frame_assignments = {}
    frame2gt_assignments = {}
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    needed_splits = 0

    start_time = time.time()

    with h5py.File(options.ground_truth_labeling, 'r') as ground_truth:
        frame_filenames_ids = [(options.new_labeling_dir + "/%04d.h5" % t, t)
                               for t in xrange(options.min_ts, options.max_ts + 1)]
        print("Checking detections from frame {} to {}".format(options.min_ts, options.max_ts))

        if options.use_compute_nodes:
            print("Setting up job cluster to dispatch work to nodes: {}".format(options.use_compute_nodes))
            import dispy
            cluster = dispy.JobCluster(check_detections_frame,
                                       nodes=options.use_compute_nodes,
                                       loglevel=logging.DEBUG)
            cluster.stats()
            results = []
        elif options.use_num_cores > 1:
            from multiprocessing import Pool
            processing_pool = Pool(processes=options.use_num_cores)
            results = []

        for filename, timestep in frame_filenames_ids:
            # get frame labels
            with h5py.File(filename, 'r') as frame:
                frame_labels = np.array(frame[options.new_label_image_path])

            # get ground truth labels for this frame
            gt_labels = np.array(ground_truth[options.gt_label_image_path][timestep:timestep + 1, ..., 0]).squeeze()

            # call detection checking, either locally, or with dispy on several nodes and processors

            if options.use_compute_nodes:
                print("Submitting job for timestep {}".format(timestep))
                print(type(gt_labels))
                print(type(frame_labels))
                job = cluster.submit(gt_labels, timestep, frame_labels, options.threshold)
                job.id = timestep
                results.append(job)
                cluster.wait()
                cluster.stats()
            elif options.use_num_cores > 1:
                async_result = processing_pool.apply_async(check_detections_frame, (gt_labels, timestep, frame_labels, options.threshold))
                results.append(async_result)
            else:
                (fn, fp, tp, ns, g2f_ass, f2g_ass, _) = check_detections_frame(gt_labels, timestep, frame_labels, options.threshold)
                false_negatives += fn
                false_positives += fp
                true_positives += tp
                needed_splits += ns
                gt2frame_assignments[timestep] = g2f_ass
                frame2gt_assignments[timestep] = f2g_ass

        # collect results if we were running in parallel
        if options.use_compute_nodes:
            cluster.wait()
            cluster.stats()
            for result in results:
                try:
                    (fn, fp, tp, ns, g2f_ass, f2g_ass, timestep) = result()
                    false_negatives += fn
                    false_positives += fp
                    true_positives += tp
                    needed_splits += ns
                    gt2frame_assignments[timestep] = g2f_ass
                    frame2gt_assignments[timestep] = f2g_ass
                    print(result.stdout)
                except:
                    print(result.stdout)
                    print(result.stderr)
                    print(result.exception)
        elif options.use_num_cores > 1:
            processing_pool.close()
            processing_pool.join()
            for result in results:
                (fn, fp, tp, ns, g2f_ass, f2g_ass, timestep) = result.get()
                false_negatives += fn
                false_positives += fp
                true_positives += tp
                needed_splits += ns
                gt2frame_assignments[timestep] = g2f_ass
                frame2gt_assignments[timestep] = f2g_ass

    end_time = time.time()
    print("\n\n------------------------------\nChecking detections took {} secs".format(end_time - start_time))

    return false_negatives, false_positives, true_positives, needed_splits, gt2frame_assignments, frame2gt_assignments


def get_assignments(assignments, timestep, idx):
    try:
        res = assignments[timestep][idx]
        if len(res) == 0:
            raise ValueError()
        return res
    except KeyError:
        raise ValueError()


def is_in_moves(src_ids, dst_ids, moves):
    if len(moves) == 0:
        return False
    for idx in src_ids:
        src_id_mappings = moves[moves[:, 0] == idx, 1]
        for mapping in src_id_mappings:
            if mapping in dst_ids:
                return True
    return False

def is_in_splits(src_ids, dst_ids, splits):
    if len(splits) == 0:
        return False
    for idx in src_ids:
        src_id_mappings = splits[splits[:, 0] == idx, 1:].squeeze()
        for mapping in src_id_mappings:
            if mapping in dst_ids:
                return True
    return False


def check_edges_frame(ground_truth, timestep, filename, options, gt2frame_assignments, frame2gt_assignments):
    print("---------------------------------\nChecking Timestep: " + str(timestep) + '\n')
    num_gt_edges = 0

    with h5py.File(filename, 'r') as frame:
        # for each ground truth move, look up whether there is the same edge in this frame's moves
        try:
            gt_moves = ground_truth["tracking"]["%04d" % timestep]["Moves"]
            gt_moves = np.array(gt_moves)
        except:
            print("Warning: Ground Truth has no Moves at {}".format(timestep))
            gt_moves = np.zeros(0)

        try:
            frame_moves = frame["tracking"]["Moves"]
            frame_moves = np.array(frame_moves)
        except:
            print("Warning: Frame {} has no Moves".format(timestep))
            frame_moves = np.zeros(0)

        try:
            gt_splits = ground_truth["tracking"]["%04d" % timestep]["Splits"]
            gt_splits = np.array(gt_splits)
        except:
            print("Warning: Ground Truth has no Splits at {}".format(timestep))
            gt_splits = np.zeros(0)

        try:
            frame_splits = frame["tracking"]["Splits"]
            frame_splits = np.array(frame_splits)
        except:
            print("Warning: Frame {} has no Splits".format(timestep))
            frame_splits = np.zeros(0)

        edges_to_add = set()
        edges_to_change = set()
        edges_to_delete_1 = set()
        edges_to_delete_2 = set()

        # handle moves
        if gt_moves.shape[0] > 0:
            num_gt_edges += len(gt_moves)

            if frame_moves.shape[0] > 0:

                # try to find matches of gt edges to frame edges
                for i in range(gt_moves.shape[0]):
                    gt_src, gt_dst = gt_moves[i]

                    try:
                        frame_src_ids = get_assignments(gt2frame_assignments, timestep - 1, gt_src)
                        frame_dst_ids = get_assignments(gt2frame_assignments, timestep, gt_dst)

                        # if any of the [src] x [dst] combinations is in frame_moves, this is good (really?)
                        # no edge between these detections in current frame -> needs to be added
                        if not is_in_moves(frame_src_ids, frame_dst_ids, frame_moves):
                            if is_in_splits(frame_src_ids, frame_dst_ids, frame_splits):
                                edges_to_change.add((frame_src_ids[0], frame_dst_ids[0]))
                            else:
                                edges_to_add.add((gt_src, gt_dst))
                    except ValueError:
                        # we did not even find a mapping of nodes, so this edge needs to be added
                        edges_to_add.add((gt_src, gt_dst))

                # see which of the frame edges that are not covered by gt are delete_1 or _2
                for i in range(frame_moves.shape[0]):
                    frame_src, frame_dst = frame_moves[i]

                    try:
                        gt_src_ids = get_assignments(frame2gt_assignments, timestep - 1, frame_src)
                        gt_dst_ids = get_assignments(frame2gt_assignments, timestep, frame_dst)

                        if not is_in_moves(gt_src_ids, gt_dst_ids, gt_moves):
                            # this edge connects two true positive detections,
                            # but the edge is not present in ground truth
                            edges_to_delete_2.add((frame_src, frame_dst))
                    except:
                        # at least one of the nodes has no mapping
                        edges_to_delete_1.add((frame_src, frame_dst))
            else:
                # all edges need to be added if there were none in this frame's move events
                for i in range(gt_moves.shape[0]):
                    edges_to_add.add((gt_moves[i, 0], gt_moves[i, 1]))

        # handle splits

        # go through all ground truth splits
        for i in range(gt_splits.shape[0]):
            num_gt_edges += 2 * gt_splits.shape[0]

            gt_src = gt_splits[i][0]

            try:
                frame_src_ids = get_assignments(gt2frame_assignments, timestep - 1, gt_src)
            except ValueError:
                # # if parent node is not detected in frame, then we need to add as many edges as there are children
                # print("Missing GT Edge: {} -> {} in timestep {}".format(gt_src, gt_dst, timestep))
                # edge_add += gt_splits.shape[1] - 1
                continue

            # look at each child node
            for x in range(1, gt_splits.shape[1]):
                # check whether it was detected at all
                try:
                    frame_dst_ids = get_assignments(gt2frame_assignments, timestep, gt_splits[i][x])
                except ValueError:
                    edges_to_add.add((gt_src, gt_splits[i][x]))
                    continue

                if not is_in_splits(frame_src_ids, frame_dst_ids, frame_splits):
                    # if we did not find that edge, check whether it is a move in the frame
                    if is_in_moves(frame_src_ids, frame_dst_ids, frame_moves):
                        edges_to_change.add((frame_src_ids[0], frame_dst_ids[0]))
                    else:
                        edges_to_add.add((gt_src, gt_splits[i][x]))


        # go through all splits in this frame
        for i in range(frame_splits.shape[0]):
            frame_src = frame_splits[i][0]

            # if parent node is not detected in frame, then we need to add as many edges as there are children
            try:
                gt_src_ids = get_assignments(frame2gt_assignments, timestep - 1, frame_src)
            except ValueError:
                print("Something weird going on!")
                # edge_add += frame_splits.shape[1] - 1
                continue

            # look at each child node
            for x in range(1, frame_splits.shape[1]):
                # check whether it was detected at all
                try:
                    gt_dst_ids = get_assignments(frame2gt_assignments, timestep, frame_splits[i][x])
                except ValueError:
                    edges_to_delete_1.add((frame_src, frame_splits[i][x]))
                    continue

                if not is_in_splits(gt_src_ids, gt_dst_ids, gt_splits):
                    if is_in_moves(gt_src_ids, gt_dst_ids, gt_moves):
                        edges_to_change.add((frame_src, frame_splits[i][x]))
                    else:
                        edges_to_delete_2.add((frame_src, frame_splits[i][x]))

        for a, b in edges_to_add:
            print("Missing GT Edge: {} -> {} in timestep {}".format(a, b, timestep))

        for a, b in edges_to_change:
            print("Wrong semantics of edge: {} -> {} in timestep {}".format(a, b, timestep))

        for a, b in edges_to_delete_1:
            print("Invalid Edge: {} -> {} in timestep {}".format(a, b, timestep))

        for a, b in edges_to_delete_2:
            print("Redundant Edge: {} -> {} in timestep {}".format(a, b, timestep))


    return num_gt_edges, len(edges_to_delete_1), len(edges_to_delete_2), len(edges_to_add), len(edges_to_change)


def check_edges(options, gt2frame_assignments, frame2gt_assignments):
    print("\n===================================\nChecking edges:\n")
    num_gt_edges = 0
    edge_delete_1 = 0
    edge_delete_2 = 0
    edge_add = 0
    edge_change = 0

    with h5py.File(options.ground_truth_labeling, 'r') as ground_truth:
        frame_filenames_ids = [(options.new_labeling_dir + "/%04d.h5" % t, t) for t in xrange(options.min_ts, options.max_ts)]
        print("Checking edges from frame {} to {}".format(options.min_ts, options.max_ts))

        for filename, timestep in frame_filenames_ids:
            ne, ed1, ed2, ea, ec = check_edges_frame(ground_truth, timestep, filename, options, gt2frame_assignments, frame2gt_assignments)
            num_gt_edges += ne
            edge_delete_1 += ed1
            edge_delete_2 += ed2
            edge_add += ea
            edge_change += ec

    return num_gt_edges, edge_delete_1, edge_delete_2, edge_add, edge_change


def compute_tra_loss(options):
    ## compute detection based values, or load old results from file
    if not options.load_detections:
        false_negatives, false_positives, true_positives, \
        needed_splits, gt2frame_assignments, frame2gt_assignments = check_detections(options)

        if options.dump_detections:
            with open(options.dump_detections, 'wb') as dump:
                pickle.dump(false_negatives, dump)
                pickle.dump(false_positives, dump)
                pickle.dump(true_positives, dump)
                pickle.dump(needed_splits, dump)
                pickle.dump(gt2frame_assignments, dump)
                pickle.dump(frame2gt_assignments, dump)
                print("Done saving!")
    else:
        with open(options.load_detections, 'rb') as dump:
            false_negatives = pickle.load(dump)
            false_positives = pickle.load(dump)
            true_positives = pickle.load(dump)
            needed_splits = pickle.load(dump)
            gt2frame_assignments = pickle.load(dump)
            frame2gt_assignments = pickle.load(dump)
            print("Done loading!")

    # determine missing edges etc
    num_gt_edges, edge_delete_1, edge_delete_2, edge_add, edge_change = check_edges(options,
                                                                                    gt2frame_assignments,
                                                                                    frame2gt_assignments)
    # compute TRA_p score
    feature_vector = [needed_splits,
                       false_negatives,
                       false_positives,
                       edge_delete_1,
                       edge_delete_2,
                       edge_add,
                       edge_change]

    weight_vector = np.array([5.0, 10.0, 1.0, 0.0, 1.0, 1.5, 1.0])
    tra_p = np.dot(np.array(feature_vector), weight_vector)

    # compute TRA_e = cost for creating GT from scratch
    num_gt_detections = true_positives + false_negatives
    tra_e = weight_vector[1] * num_gt_detections + weight_vector[5] * num_gt_edges

    print("\n===================================\nFinal Results:\n")
    print("Needed Splits: {} (weight={})".format(needed_splits, weight_vector[0]))
    print("False Negatives: {} (weight={})".format(false_negatives, weight_vector[1]))
    print("False Positives: {} (weight={})".format(false_positives, weight_vector[2]))
    print("Edge Delete 1: {} (weight={})".format(edge_delete_1, weight_vector[3]))
    print("Edge Delete 2: {} (weight={})".format(edge_delete_2, weight_vector[4]))
    print("Edge Add: {} (weight={})".format(edge_add, weight_vector[5]))
    print("Edge Change: {} (weight={})".format(edge_change, weight_vector[6]))

    print("Extracted events yield tra_p={} and tra_e={}".format(feature_vector, tra_p, tra_e))
    return min(tra_p, tra_e) / tra_e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute TRA loss of a new labeling compared to ground truth')

    # file paths
    parser.add_argument('--ground-truth-labeling', required=True, type=str, dest='ground_truth_labeling',
                        help='HDF5 file containing the ground truth labeling')
    parser.add_argument('--gt-label-image-path', type=str, dest='gt_label_image_path', default='exported_data',
                        help='Path to the label image inside the ground truth HDF5 file')
    parser.add_argument('--new-labeling-dir', required=True, type=str, dest='new_labeling_dir',
                        help='Folder containing the HDF5 frame by frame results of a new labeling [default=%default]')
    parser.add_argument('--new-label-image-path', type=str, dest='new_label_image_path', default='segmentation/labels',
                        help='Path to the label image inside the new HDF5 files [default=%default]')
    parser.add_argument('--dump-detections', type=str, dest='dump_detections', default=None,
                        help='File where to dump the results of checking the detections.')
    parser.add_argument('--load-detections', type=str, dest='load_detections', default=None,
                        help='File where to load dumped results of checking the detections.')

    # time range
    parser.add_argument('--min-ts', type=int, dest='min_ts', default=0,
                        help='First timestep to look at [default=%default]')
    parser.add_argument('--max-ts', type=int, dest='max_ts', default=-1,
                        help='Last timestep to look at (not inclusive!) [default=%default]')

    # detection acceptance threshold (Jaccard index)
    parser.add_argument('--threshold', type=float, dest='threshold', default=0.5,
                        help='Jaccard index at which overlap ratio a detection counts as true positive[default=%default]')

    # parallelization
    parser.add_argument('--use-compute-nodes', type=str, nargs='+', dest='use_compute_nodes',
                        help='List of IP adresses where dispynode.py is running')
    parser.add_argument('--use-num-cores', type=int, dest='use_num_cores', default=1,
                        help='Use multiprocessing with the given num of cores (>1!). '
                             'Can not be used simultaneously with compute-nodes!')

    # parse command line
    args = parser.parse_args()

    # make sure time range is okay

    with h5py.File(args.ground_truth_labeling, 'r') as ground_truth:
        shape = ground_truth[args.gt_label_image_path].shape
        print("Found ground truth label image of shape {}".format(shape))

        if args.max_ts < 0:
            args.max_ts += shape[0]
        args.max_ts = min(shape[0], args.max_ts)

    loss = compute_tra_loss(args)
    print("found loss value of: {}".format(loss))