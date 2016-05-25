#!/usr/bin/env python
import os
import os.path as path
import sys
sys.path.append('../.')
sys.path.append('.')

import imp
multitrack = imp.load_source("multitrack", "./multitrack_ilastik10")
import optparse
import time
import glob
import numpy as np
import h5py
import itertools
import vigra
import pgmlink as track
from multiprocessing import Pool

def rank_solutions(options):
    assert(len(options.reranker_weight_file) > 0)
    feature_vectors = np.loadtxt(options.out_dir.rstrip('/') + '/feature_vectors.txt')
    weights = np.loadtxt(options.reranker_weight_file)

    scores = np.dot(feature_vectors, weights)
    print("Reranker Assigned window scores: ".format(scores))
    return np.argmax(scores)

def track_subgraphs(graph,
                    time_range,
                    timesteps_per_segment,
                    segment_overlap_timesteps,
                    conservation_tracking_parameter,
                    fov,
                    ilp_fn,
                    ts, 
                    fs,
                    t0,
                    trans_classifier,
                    uncertaintyParam
                    ):
    """
    Experiment: track only subgraphs of the full hypotheses graph with some overlap,
    and then stitch the results together using fusion moves.
    """
    # define which segments we have
    num_segments = int(np.ceil(float((time_range[1] - time_range[0])) / (timesteps_per_segment - segment_overlap_timesteps)))
    segments = [(time_range[0] + i * (timesteps_per_segment - segment_overlap_timesteps),
                 (time_range[0] + (i + 1) * timesteps_per_segment - i * segment_overlap_timesteps))
                for i in xrange(num_segments)]

    tmap = graph.getNodeTraxelMap()
    solutions = {}
    arc_solutions = {}
    div_solutions = {}

    original_out_dir = options.out_dir

    # track all segments individually
    for i, segment in enumerate(segments):
        print("************** Creating subgraph for timesteps in {}".format(segment))

        # use special out-dir per window
        options.out_dir = original_out_dir.rstrip('/') + '/window_' + str(i) + '/'
        try:
            os.makedirs(options.out_dir)
        except:
            pass

        # create subgraph for this segment
        node_mask = track.NodeMask(graph)
        n_it = track.NodeIt(graph)
        for n in n_it:
            node_mask[n] = segment[0] <= tmap[n].Timestep < segment[1]

        arc_mask = track.ArcMask(graph)
        a_it = track.ArcIt(graph)
        for a in a_it:
            arc_mask[a] = tmap[graph.source(a)].Timestep >= segment[0] and tmap[graph.target(a)].Timestep < segment[1]
        subgraph = track.HypothesesGraph()
        track.copy_hypotheses_subgraph(graph, subgraph, node_mask, arc_mask)
        subgraph_node_origin_map = subgraph.getNodeOriginReferenceMap()
        subgraph_arc_origin_map = subgraph.getArcOriginReferenceMap()
        subgraph.initLabelingMaps()

        # fix variables in overlap
        if i > 0:
            sub_tmap = subgraph.getNodeTraxelMap()
            n_it = track.NodeIt(subgraph)
            for n in n_it:
                if segment[0] == sub_tmap[n].Timestep:
                    origin_node = subgraph_node_origin_map[n]
                    origin_node_id = graph.id(origin_node)
                    subgraph.addAppearanceLabel(n, solutions[origin_node_id][-1])
                    print "fixing node ", origin_node_id, " which is ", subgraph.id(n), " in subgraph"

        print("Subgraph has {} nodes and {} arcs".format(track.countNodes(subgraph), track.countArcs(subgraph)))

        # create subgraph tracker
        subgraph_tracker = track.ConsTracking(subgraph,
                                                ts,
                                                conservation_tracking_parameter,
                                                uncertaintyParam,
                                                fov,
                                                bool(options.size_dependent_detection_prob),
                                                options.avg_obj_size[0],
                                                options.mnd,
                                                options.division_threshold)
        all_events = subgraph_tracker.track(conservation_tracking_parameter, bool(i > 0))

        if len(options.raw_filename) > 0 and len(options.reranker_weight_file) > 0:
            # run merger resolving and feature extraction, which also returns the score of each proposal
            region_features = multitrack.getRegionFeatures(ndim)
            scores = multitrack.runMergerResolving(options, 
                subgraph_tracker, 
                ts,
                fs,
                subgraph,
                ilp_fn,
                all_events,
                fov,
                region_features,
                trans_classifier,
                segment[0],
                True)

            best_sol_idx = int(np.argmax(np.array(scores)))
            subgraph.set_solution(best_sol_idx)
            print("====> selected solution {} in window {} <=====".format(best_sol_idx, i))
        else:
            subgraph.set_solution(0)
        print("Done tracking subgraph")

        # collect solutions
        subgraph_node_active_map = subgraph.getNodeActiveMap()
        subgraph_arc_active_map = subgraph.getArcActiveMap()
        subgraph_div_active_map = subgraph.getDivisionActiveMap()

        n_it = track.NodeIt(subgraph)
        for n in n_it:
            origin_node = subgraph_node_origin_map[n]
            origin_node_id = graph.id(origin_node)
            value = subgraph_node_active_map[n]

            if not origin_node_id in solutions:
                solutions[origin_node_id] = [value]
            else:
                solutions[origin_node_id].append(value)
            div_solutions[origin_node_id] = subgraph_div_active_map[n]
        a_it = track.ArcIt(subgraph)
        for a in a_it:
            origin_arc = subgraph_arc_origin_map[a]
            origin_arc_id = graph.id(origin_arc)
            arc_solutions[origin_arc_id] = subgraph_arc_active_map[a]
        print("Done storing solutions")

    # reset out-dir
    options.out_dir = original_out_dir

    # find overlapping variables
    print("Computing overlap statistics...")
    num_overlap_vars = sum([1 for values in solutions.values() if len(values) > 1])
    num_disagreeing_overlap_vars = sum([1 for values in solutions.values() if len(values) > 1 and values[0] != values[1]])

    for key, values in solutions.iteritems():
        if len(values) > 1 and values[0] != values[1]:
            print("\tFound disagreement at {}: {} != {}".format(key, values[0], values[1]))

    print("Found {} variables in overlaps, of which {} did disagree ({}%)".format(num_overlap_vars,
                                                                                  num_disagreeing_overlap_vars,
                                                                                  100.0 * float(num_disagreeing_overlap_vars) / num_overlap_vars))
    
    if num_disagreeing_overlap_vars == 0:
        # write overall solution back to hypotheses graph
        graph.initLabelingMaps()
        n_it = track.NodeIt(graph)
        for n in n_it:
            n_id = graph.id(n)

            graph.addAppearanceLabel(n, solutions[n_id][-1])
            graph.addDisappearanceLabel(n, solutions[n_id][-1])

            # store division information
            graph.addDivisionLabel(n, div_solutions[n_id])

        # activate arcs
        a_it = track.ArcIt(graph)
        for a in a_it:
            a_id = graph.id(a)
            graph.addArcLabel(a, arc_solutions[a_id])
        graph.set_injected_solution()
    else:
        raise AssertionError("Nodes did disagree, cannot create stitched solution")

if __name__ == "__main__":
    options, args = multitrack.getConfigAndCommandLineArguments()

    # get filenames
    numArgs = len(args)
    fns = []
    if numArgs > 0:
        for arg in args:
            print arg
            fns.extend(glob.glob(arg))
        fns.sort()
        print(fns)

    print fns
    ilp_fn = fns[0]

    # create output path
    if not path.exists(options.out_dir):
        try:
            os.makedirs(options.out_dir)
        except:
            pass

    ### Do the tracking
    start = time.time()

    feature_path = options.feats_path
    with_div = True
    with_merger_prior = True

    # get selected time range
    time_range = [options.mints, options.maxts]
    if options.maxts == -1 and options.mints == 0:
        time_range = None

    trans_classifier = multitrack.loadTransClassifier(options)

    # set average object size if chosen
    obj_size = [0]
    if options.avg_obj_size != 0:
        obj_size[0] = options.avg_obj_size
    else:
        options.avg_obj_size = obj_size

    # find shape of dataset
    with h5py.File(ilp_fn, 'r') as h5file:
        shape = h5file['/'.join(options.label_img_path.split('/')[:-1])].values()[0].shape[1:4]

    # read all traxels into TraxelStore
    ts, fs, max_traxel_id_at, ndim, t0, t1 = multitrack.getTraxelStore(options, ilp_fn, time_range, shape)

    print "Start tracking..."
    if options.method != "conservation" and options.method != 'conservation-dynprog':
        raise Exception("unknown tracking method: " + options.method)

    w_det, w_div, w_trans, w_dis, w_app, = (options.detection_weight, options.division_weight, options.transition_weight,
        options.disappearance_cost, options.appearance_cost,)

    # generate structured learning files
    if options.export_funkey and len(options.gt_pth) > 0:
        tracker, fov = multitrack.initializeConservationTracking(options, shape, t0, t1)
        outpath = multitrack.exportFunkeyFiles(options, ts, tracker, trans_classifier)

        if options.only_labels:
            print "finished writing labels to "+outpath
            exit(0)

        if options.learn_funkey:
            learned_weights = multitrack.learnFunkey(options, tracker, outpath)
            w_det, w_div, w_trans, w_dis, w_app, = learned_weights

        if options.learn_perturbation_weights:
            multitrack.learn_perturbation_weights(ts, options, shape, trans_classifier, t0, t1)
            exit(0)

    # higher order feature file preparation
    feature_vector_filename = options.out_dir.rstrip('/') + '/ho_feature_vectors.txt'
    try:
        os.remove(feature_vector_filename)
    except:
        pass

    # -------------------------------------------------------
    # perform the real tracking
    tracker, fov = multitrack.initializeConservationTracking(options, shape, t0, t1)

    # train outlier svm if needed
    if len(options.save_outlier_svm) > 0 and len(options.gt_pth) > 0:
        multitrack.train_outlier_svm(options, tracker, ts, fov)
    
    if options.num_iterations == 0:
        exit(0)
    
    # build hypotheses graph
    print "tracking with weights ", w_det, w_div, w_trans, w_dis, w_app
    hypotheses_graph = tracker.buildGraph(ts)

    if options.w_labeling:
        assert len(options.gt_pth) > 0, 'if labeling should be loaded, please provide a path to the ground truth in --gt-pth'
        multitrack.store_label_in_hypotheses_graph(options, hypotheses_graph, tracker)

    # perturbation settings
    uncertaintyParam = multitrack.getUncertaintyParameter(options)
    if options.num_iterations == 0 or options.num_iterations == 1 or options.save_first:
        proposal_out = ''
    else:
        proposal_out = options.out_dir.rstrip('/') + '/proposal_labeling'
        
    
    tracker.setTrackLabelingExportFile(proposal_out)

    solver = track.ConsTrackingSolverType.CplexSolver
    if options.method == 'conservation-dynprog':
        solver = track.ConsTrackingSolverType.DynProgSolver

     # ---------------------------------------------------
    # track with subgraphs
    conservation_tracking_parameter = tracker.get_conservation_tracking_parameters(
        options.forbidden_cost,
        options.ep_gap,
        not bool(options.without_tracklets),
        w_det,
        w_div,
        w_trans,
        w_dis,
        w_app,
        options.with_merger_resolution,
        ndim,
        options.trans_par,
        options.border_width,
        not bool(options.woconstr),
        uncertaintyParam,
        options.timeout,
        trans_classifier,
        solver
    )

    track_subgraphs(hypotheses_graph, 
        (t0, t1), 
        10, # timesteps_per_segment,
        1, # segment_overlap_timesteps,
        conservation_tracking_parameter, 
        fov, 
        ilp_fn,
        ts, 
        fs,
        t0,
        trans_classifier,
        uncertaintyParam)
    all_events = [track.getEventsOfGraph(hypotheses_graph)]
    # # track!
    # all_events = tracker.track(
    #     options.forbidden_cost,
    #     options.ep_gap,
    #     not bool(options.without_tracklets),
    #     w_det,
    #     w_div,
    #     w_trans,
    #     w_dis,
    #     w_app,
    #     options.with_merger_resolution,
    #     ndim,
    #     options.trans_par,
    #     options.border_width,
    #     not bool(options.woconstr),
    #     uncertaintyParam,
    #     options.timeout,
    #     trans_classifier, # pointer to transition classifier
    # )

    tracker.setTrackLabelingExportFile('')

    # run merger resolving
    if options.with_merger_resolution and options.max_num_objects > 1 and len(options.raw_filename) > 0:
        region_features = multitrack.getRegionFeatures(ndim)
        # try:
        multitrack.runMergerResolving(options, tracker, ts, fs, hypotheses_graph, ilp_fn, all_events, fov, region_features, trans_classifier, t0, True)
        # except BaseException as e:
            # print("WARNING: Merger Resolving crashed...: {}".format(e))

    stop = time.time()
    since = stop - start

    if options.w_labeling:
        hypotheses_graph.write_hypotheses_graph_state(options.out_dir.rstrip('/') + '/results_all.txt')
    
    # save
    if not options.skip_saving:
        events = all_events[0]
        first_events = events[0]
        events = events[1:]
        out_dir = options.out_dir.rstrip('/') + '/iter_0'
        multitrack.save_events(out_dir, events, shape, t0, t1,
                      options.label_img_path, max_traxel_id_at, ilp_fn, first_events)

    print "Elapsed time [s]: " + str(int(since))
    print "Elapsed time [min]: " + str(int(since) / 60)
    print "Elapsed time [h]: " + str(int(since) / 3600)