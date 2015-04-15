import sys
sys.path.append('../.')
sys.path.append('.')
import h5py
import pgmlink as track
import numpy as np
import math
import os
import os.path as path
import argparse

class ProgressBar:
    def __init__(self, start=0, stop=100):
        self._state = 0
        self._start = start
        self._stop = stop

    def reset(self, val=0):
        self._state = val

    def show(self, increase=1):
        self._state += increase
        if self._state > self._stop:
            self._state = self._stop

        # show
        pos = float(self._state - self._start)/(self._stop - self._start)
        sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*pos), (100*pos)))

        if self._state == self._stop:
            sys.stdout.write('\n')
        sys.stdout.flush()


def get_feature(h5file, feature_path, object_id):
    if not feature_path in h5file:
        raise Exception("Feature %s not found in %s." % (feature_path, h5file.filename))
    return h5file[feature_path][object_id, ...]


def get_feature_vector(traxel, feature_name, num_dimensions):
    #print traxel.print_available_features()
    result = []
    for i in range(num_dimensions):
        try:
            result.append(traxel.get_feature_value(feature_name, i))
        except:
            print("Error when accessing feature {}[{}] for traxel (Id={},Timestep={})".format(feature_name,
                                                                                              i,
                                                                                              traxel.Id,
                                                                                              traxel.Timestep))
            print "Available features are: "
            print traxel.print_available_features()
            raise Exception
    return result


def rank_solutions(ground_truth_filename, feature_vector_filename, proposal_basename, num_iterations):
    import structsvm
    proposal_filenames = [proposal_basename + '_' + str(i) + '.txt' for i in range(num_iterations)]

    problem = structsvm.struct_svm_ranking_problem(feature_vector_filename,
                                                   ground_truth_filename,
                                                   proposal_filenames)
                                                   #,args.loss_weights)
    solver = structsvm.struct_svm_solver_primal(problem)
    weights = solver.solve()
    weights = np.array(weights)
    problem.print_scores(weights)

    print("Found weights: {}".format(weights))
    return weights


class LineagePart:
    """
    Parent class for all parts of a lineage, e.g. track, division, lineage
    """

    def __init__(self):
        self.features = {}
        pass

    track_feature_map = {
        'single': [
            'track_outlier_svm_score', # TODO: add length?
        ],
        'mean_var': [
            'sq_diff_RegionCenter',
            'sq_diff_Count',
            'sq_diff_Mean',
            'sq_diff_Variance',
            'sq_accel_RegionCenter',
            'sq_accel_Count',
            'sq_accel_Mean',
            'sq_accel_Variance',
            'angles_RegionCenter',
            'outlier_id_RegionCenter',
            'outlier_id_Count',
            'outlier_id_Mean',
            'outlier_id_Variance',
            'diff_outlier_RegionCenter',
            'diff_outlier_Count',
            'diff_outlier_Mean',
            'diff_outlier_Variance',
            'Count',
            'Mean',
            'Variance',
            'Sum',
            'Kurtosis',
            'Skewness',
            'Weighted<PowerSum<0> >',
            'Central< PowerSum<2> >',
            'Central< PowerSum<3> >',
            'Central< PowerSum<4> >'
        ]
    }

    division_feature_map = {
        'single': [
            'div_outlier_svm_score'
        ],
        'mean_var': [
            'child_decel_Count',
            'child_decel_Count_outlier_score',
            'child_decel_Mean',
            'child_decel_Mean_outlier_score',
            'child_decel_RegionCenter',
            'child_decel_RegionCenter_outlier_score',
            'child_decel_Variance',
            'child_decel_Variance_outlier_score',
            'sq_diff_Count',
            'sq_diff_Count_outlier_score',
            'sq_diff_Mean',
            'sq_diff_Mean_outlier_score',
            'sq_diff_RegionCenter',
            'sq_diff_RegionCenter_outlier_score',
            'sq_diff_Variance',
            'sq_diff_Variance_outlier_score'
        ]
    }

    all_feature_names = track_feature_map['single'] \
                        + ['mean_' + f for f in track_feature_map['mean_var']] \
                        + ['var_' + f for f in track_feature_map['mean_var']] \
                        + division_feature_map['single'] \
                        + ['mean_' + f for f in division_feature_map['mean_var']] \
                        + ['var_' + f for f in division_feature_map['mean_var']]

    @staticmethod
    def get_num_track_features():
        return 2 * len(LineagePart.track_feature_map['mean_var']) + len(LineagePart.track_feature_map['single'])

    @staticmethod
    def get_num_division_features():
        return 2 * len(LineagePart.division_feature_map['mean_var']) + len(LineagePart.division_feature_map['single'])

    @staticmethod
    def get_feature_vector_size():
        return LineagePart.get_num_track_features() + LineagePart.get_num_division_features()

    @staticmethod
    def weight_idx_to_feature(idx):
        return LineagePart.all_feature_names[idx]

    @staticmethod
    def feature_to_weight_idx(feature_name):
        return LineagePart.all_feature_names.index(feature_name)

    @staticmethod
    def get_expanded_feature_names(series_expansion_range):
        len_track_feat = LineagePart.get_num_track_features()

        expanded_feature_names = []
        for f in LineagePart.all_feature_names[:len_track_feat]:
            for i in range(series_expansion_range[0], series_expansion_range[1]):
                exponent = str(i) + " - "
                expanded_feature_names += [exponent + f, ]

        expanded_feature_names += LineagePart.all_feature_names[len_track_feat:]
        return expanded_feature_names

    def get_feature_vector(self):
        """
        This should return the feature vector (concatenated track and division features)
        such that it can be multiplied by the weight vector to return this track's score
        :return: a numpy array
        """
        result = np.zeros(LineagePart.get_feature_vector_size())

        for i, k in enumerate(LineagePart.all_feature_names):
            if k in self.features:
                result[i] = self.features[k]

        return result

    def get_expanded_feature_vector(self, series_expansion_range):
        """
        To allow for using length scaled versions of the track features,
        this vector contains modified duplicates the track features,
        followed by the division features.
        The modification can be specified by overwriting the expansion_modifier function
        :param series_expansion_range: (lower_bound, upper_bound) of the expansion range (used as exponent)
        :return:
        """
        num_expansions = series_expansion_range[1] - series_expansion_range[0]
        len_track_feat = LineagePart.get_num_track_features()

        result = np.zeros(num_expansions * len_track_feat + LineagePart.get_num_division_features())

        # insert duplicated and modified track features
        # (from [f1, f2, f3] to [f'1, f''1, f'''1, f'2, f''2, f'''2, f'3, f''3, f'''3]
        for i, k in enumerate(LineagePart.all_feature_names[:len_track_feat]):
            if k in self.features:
                for e in range(num_expansions):
                    result[i * num_expansions + e] = self.features[k] * self.expansion_factor(e)

        # insert division features just once
        for i, k in enumerate(LineagePart.all_feature_names[len_track_feat:]):
            if k in self.features:
                result[num_expansions * len_track_feat + i] = self.features[k]

        return result

    def expansion_factor(self, expansion):
        """
        Expansion modifier, defaults to identity.
        :param expansion:
        :return:
        """
        return 1

    def extract(self, track_features_h5):
        """
        Extract all important information about this part of the lineage from the features h5 file
        """
        raise NotImplementedError("Please Specialize this method")

    def compute_score(self, weights):
        """
        :return: the score as linear combination of feature vector and supplied weights
        """
        feature_vec = self.get_feature_vector()
        return np.dot(weights, feature_vec)


class Track(LineagePart):
    """
    A track is a part of a lineage tree between two division/appearance/disappearance events
    """

    def __init__(self, track_id):
        LineagePart.__init__(self)
        self.track_id = track_id
        # number of traxels in track
        self.length = 0

        # reference to division that created this track. -1 if it stems from appearance
        self.start_division_id = -1
        # reference to division where this track ends. -1 if it disappears
        self.end_division_id = -1

        # first and last traxel ids
        self.start_traxel_id = -1
        self.end_traxel_id = -1
        self.traxels = None

        # map of named features of this track
        self.features = {}

    def extract_region_features(self, ts, region_features):
        """
        Extract the region features of this track's traxels from the traxelstore,
        and compute mean an variance.
        """
        assert(self.traxels != None and len(self.traxels) > 0)
        
        # temporary storage for features over traxels
        f = {}

        for timestep, idx in self.traxels.transpose():
            traxel = ts.get_traxel(int(idx), int(timestep))
            for feat_name, feat_dims in region_features:
                if feat_name not in f:
                    f[feat_name] = []
                f[feat_name].append(get_feature_vector(traxel, feat_name, feat_dims))

        # create a feature matrix per feature, and store mean + var
        for k in LineagePart.track_feature_map['mean_var']:
            if k in f:
                v = f[k]
                feature_matrix = np.array(v)
                # compute mean & variance
                self.features['mean_' + k] = np.mean(feature_matrix)
                self.features['var_'  + k] = np.var(feature_matrix)

        # get the feature value for the arguments where mean + var doesn't make sense
        for k in LineagePart.track_feature_map['single']:
            # split multi-dim-features
            if '[' in k:
                try:
                    k, idx = k.split('[')
                    idx = int(idx.replace(']', ''))
                except:
                    print "Did not recognize format of feature name: ", k
                    idx = 0
            else:
                idx = 0

            if k in f:
                feature_matrix = np.array(f[k]).flatten()
                self.features[k] = feature_matrix[idx]

    def extract(self, track_features_h5):
        """
        Get the higher order features of this track from the HDF5 file,
        and compute the mean of each feature.
        """
        t_features = track_features_h5['tracks/' + str(self.track_id) + '/']
        self.length = int(t_features['track_length'].value[0, 0])
        self.traxels = t_features['traxels'].value
        assert(self.traxels.shape[1] > 0)
        self.start_traxel_id = tuple(self.traxels[:, 0])
        self.end_traxel_id = tuple(self.traxels[:, -1])

        # get all features from the HDF5 file
        for v in LineagePart.track_feature_map['mean_var']:
            try:
                feature_matrix = t_features[v].value
                self.features['mean_' + v] = np.mean(feature_matrix)
                self.features['var_' + v] = np.var(feature_matrix)
            except:
                # print("Could not find feature {} for track {}".format(v, self.track_id))
                pass
        try:
            self.features['track_outlier_svm_score'] = track_features_h5['track_outliers_svm'].value.flatten()[self.track_id]
        except:
            self.features['track_outlier_svm_score'] = -1

    def expansion_factor(self, expansion):
        """
        Expansion modifier for tracks is to multiply by powers of the length
        :param expansion:
        :return:
        """
        return math.pow(self.length, expansion)


class Division(LineagePart):
    def __init__(self, division_id):
        LineagePart.__init__(self)
        self.division_id = division_id

        # track ids
        self.parent_track_id = -1
        self.children_track_ids = [-1, -1]

        # traxel ids
        self.parent_traxel_id = -1
        self.children_traxel_ids = [-1, -1]

        # set of features of this division
        self.features = {}

    def extract(self, track_features_h5):
        d_features = track_features_h5['divisions/' + str(self.division_id) + '/']
        traxels = d_features['traxels'].value
        assert(traxels.shape[1] == 3)
        self.parent_traxel_id = tuple(traxels[:, 0])
        self.children_traxel_ids = [tuple(traxels[:, 1]), tuple(traxels[:, 2])]

        # get all features from the HDF5 file
        for v in LineagePart.division_feature_map['mean_var']:
            try:
                feature_matrix = d_features[v].value
                self.features['mean_' + v] = np.mean(feature_matrix)
                self.features['var_' + v] = np.var(feature_matrix)
            except:
                # print("Could not find feature {} for division {}".format(v, self.division_id))
                pass

        try:
            self.features['div_outlier_svm_score'] = track_features_h5['division_outliers_svm'].value.flatten()[self.division_id]
        except:
            self.features['div_outlier_svm_score'] = -1


class LineageTree(LineagePart):
    """
    A lineage tree holds all tracks and divisions that have one common ancestor.
    The returned feature vectors are the averaged features over tracks and divisions whithin the lineage tree.
    """

    def __init__(self, lineage_tree_id, track, tracks, divisions):
        LineagePart.__init__(self)
        self.lineage_tree_id = lineage_tree_id
        self.tracks = [track]
        self.divisions = []
        self.length = 0 # todo remove this, just left in to allow for unpickling

        # follow the supplied track along divisions
        from collections import deque
        queue = deque([track])

        while len(queue) > 0:
            t = queue.popleft()
            if t.end_division_id != -1:
                self.divisions.append(divisions[t.end_division_id])
                for i in [0, 1]:
                    next_track_id = self.divisions[-1].children_track_ids[i]
                    if next_track_id == -1:
                        print("Warning: lineage tree could not find child {} of division {}, "
                              "discarding branch.".format(i, self.divisions[-1].division_id))
                        continue
                    self.tracks.append(tracks[next_track_id])
                    queue.append(tracks[next_track_id])

    def get_feature_vector(self):
        result = np.zeros(self.get_feature_vector_size())
        
        for t in self.tracks:
            result += t.get_feature_vector() / len(self.tracks)

        for d in self.divisions:
            result += d.get_feature_vector() / len(self.divisions)

        return result

    def get_expanded_feature_vector(self, series_expansion_range):
        num_expansions = series_expansion_range[1] - series_expansion_range[0]
        result = np.zeros(num_expansions * self.get_num_track_features() + self.get_num_division_features())

        for t in self.tracks:
            result += t.get_expanded_feature_vector(series_expansion_range) / len(self.tracks)

        for d in self.divisions:
            result += d.get_expanded_feature_vector(series_expansion_range) / len(self.divisions)

        return result

    def get_all_traxels(self):
        """
        Assuming that no divisions start with an appearance, or end at disappearances,
        all participating traxels are stored inside the tracks.
        The traxels are returned as list tuples of (timestep, ID).
        """
        traxels = []
        for t in self.tracks:
            for i in xrange(t.traxels.shape[1]):
                traxels.append(tuple(t.traxels[0:2, i]))
        return traxels


def create_and_link_tracks_and_divisions(track_features_h5, ts, region_features):
    # storage for all tracks and divisions
    tracks = {}
    divisions = {}

    # mapping of traxelID at front and back of track to track_id
    track_starts_with_traxel_id = {}
    track_ends_with_traxel_id = {}

    pb = ProgressBar(0, len(track_features_h5['tracks'].keys()) + len(track_features_h5['divisions'].keys()))
    print("Extracting Tracks and Divisions")

    for track_id in track_features_h5['tracks'].keys():
        pb.show()
        track_id_int = int(track_id)
        t = Track(track_id_int)
        t.extract(track_features_h5)
        t.extract_region_features(ts, region_features)

        # store in container
        tracks[track_id_int] = t
        
        # create mappings
        track_starts_with_traxel_id[t.start_traxel_id] = track_id_int
        track_ends_with_traxel_id[t.end_traxel_id] = track_id_int

    for division_id in track_features_h5['divisions'].keys():
        pb.show()
        division_id_int = int(division_id)
        d = Division(division_id_int)
        d.extract(track_features_h5)

        # find the tracks that are connected by this division
        # and update information in the tracks
        try:
            d.parent_track_id = track_ends_with_traxel_id[d.parent_traxel_id]
            tracks[d.parent_track_id].end_division_id = division_id_int
        except KeyError as e:
            print("Could not find parent track of division {}: ".format(division_id), e.message)

        for i in [0, 1]:
            try:
                d.children_track_ids[i] = track_starts_with_traxel_id[d.children_traxel_ids[i]]
                tracks[d.children_track_ids[i]].start_division_id = division_id_int
            except KeyError as e:
                print("Could not find child track of division {}: ".format(division_id), e.message)

        # store in container
        divisions[division_id_int] = d
    return tracks, divisions


def build_lineage_trees(tracks, divisions):
    # select all tracks that have no parent division (appearances)
    appearances = [t for t in tracks.values() if t.start_division_id == -1]

    # create a new lineage tree for each of those
    lineageTrees = [LineageTree(l_id, t, tracks, divisions) for l_id, t in enumerate(appearances)]

    return lineageTrees


def score_solutions(tracks, divisions, lineage_trees, out_dir, reranker_weight_filename):
    # if reranker weights are already given, compute overall, and track scores and plot a histogram
    if reranker_weight_filename and len(reranker_weight_filename) > 0:
        reranker_weights = np.loadtxt(reranker_weight_filename)

        track_scores = [t.compute_score(reranker_weights) for t in tracks.values()]
        division_scores = [d.compute_score(reranker_weights) for d in divisions.values()]
        lineage_scores = [l.compute_score(reranker_weights) for l in lineage_trees]
        overall_score = sum(lineage_scores)

        for s_name, scores in [("track_scores", track_scores), 
                               ("division_scores", division_scores), 
                               ("lineage_scores", lineage_scores)]:
            fn = out_dir.rstrip('/') + '/' + s_name + '.txt'
            print("Saving {} to:".format(s_name, fn))
            np.savetxt(fn, scores)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(scores, 100)
            plt.savefig(out_dir.rstrip('/') + '/' + s_name + '.pdf')

            # todo: compute loss and/or precision of each track / division / lineage and plot the correspondance!

        print("Overall score of tracking is: {}".format(overall_score))
        return overall_score
    else:
        return -1


def compute_traxel_set_measures(args):
    (traxel_set, associations, filename_pairs, timestep0) = args
    from empryonic.learning import quantification as quant
    taxonomies = []

    for i, v in enumerate(filename_pairs[1:]):
        t = quant.compute_filtered_taxonomy(associations[i],
                                            associations[i + 1],
                                            v[0],
                                            v[1],
                                            traxel_set,
                                            i + timestep0 + 1)
        taxonomies.append(t)
    overall = reduce(quant.Taxonomy.union, taxonomies)
    return overall


def compare_lineage_trees_to_gt(gt_dir, proposal_dir, lineage_trees):
    import compare_tracking
    from multiprocessing import Pool, cpu_count
    import itertools

    print("Parallelizing over {} cores".format(cpu_count()))
    processing_pool = Pool(cpu_count())

    gt_filenames, proposal_filenames = compare_tracking.get_tracking_filenames(gt_dir, proposal_dir)
    first_timestep = int(os.path.splitext(os.path.basename(gt_filenames[0]))[0])

    timesteps = min(len(gt_filenames), len(proposal_filenames))
    associations = compare_tracking.construct_associations(gt_filenames, proposal_filenames, timesteps)

    filename_pairs = zip(gt_filenames[0:timesteps], proposal_filenames[0:timesteps])
    lineage_tree_traxels = [lt.get_all_traxels() for lt in lineage_trees]

    pb = ProgressBar(0, len(lineage_trees))
    lineage_tree_measures = []
    for measure in processing_pool.imap(compute_traxel_set_measures,
                                        itertools.izip(lineage_tree_traxels,
                                                       itertools.repeat(associations),
                                                       itertools.repeat(filename_pairs),
                                                       itertools.repeat(first_timestep))):
        lineage_tree_measures.append(measure)
        pb.show()

    return lineage_tree_measures


def extract_features_and_compute_score(reranker_weight_filename,
                                       outlier_svm_filename,
                                       out_dir,
                                       iteration,
                                       hypotheses_graph,
                                       ts,
                                       fov,
                                       feature_vector_filename,
                                       region_features,
                                       series_expansion_range=[-1, 2]):
    
    # extract higher order features and per-track features
    print("Extracting features of solution {}\n\tCreating Feature extractor...".format(iteration))
    track_features_filename = out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_features.h5'
    feature_extractor = track.TrackingFeatureExtractor(hypotheses_graph, fov)
    
    # load outlier SVMs if available
    if len(outlier_svm_filename) > 0:  # when the svm was trained in this run, it automatically sets load_outlier_svm
        with open(outlier_svm_filename, 'r') as svm_dump:
            import pickle
            outlier_track_svm = pickle.load(svm_dump)
            outlier_division_svm = pickle.load(svm_dump)

            feature_extractor.set_track_svm(outlier_track_svm)
            feature_extractor.set_division_svm(outlier_division_svm)
            print("Trained outlier SVM loaded from " + outlier_svm_filename)

    # extract all the features and save them to disk
    try:
        os.remove(track_features_filename)
    except:
        pass
    feature_extractor.set_track_feature_output_file(track_features_filename)
    sys.stdout.write("\tComputing features...")
    feature_extractor.compute_features()
    print("...Done")

    # create complete lineage trees with extracted features:
    with h5py.File(track_features_filename, 'r') as track_features_h5:
        tracks, divisions = create_and_link_tracks_and_divisions(track_features_h5, ts, region_features)
        lineage_trees = build_lineage_trees(tracks, divisions)

    # save lineage trees
    lineage_tree_dump_filename = out_dir.rstrip('/') + '/iter_' + str(iteration) + '/lineage_trees.dump'
    save_lineage_dump(lineage_tree_dump_filename, tracks, divisions, lineage_trees)

    # accumulate feature vectors
    feature_vector = lineage_trees[0].get_expanded_feature_vector(series_expansion_range)
    for lt in lineage_trees[1:]:
        feature_vector += lt.get_expanded_feature_vector(series_expansion_range)

    # compute score if weight file is given
    overall_score = score_solutions(tracks, divisions, lineage_trees,
                                    out_dir.rstrip('/') + '/iter_' + str(iteration), reranker_weight_filename)

    return feature_vector, overall_score, LineagePart.all_feature_names


def load_lineage_dump(filename):
    with open(filename, 'r') as lineage_dump:
        import pickle
        tracks = pickle.load(lineage_dump)
        divisions = pickle.load(lineage_dump)
        lineage_trees = pickle.load(lineage_dump)

    return tracks, divisions, lineage_trees


def save_lineage_dump(filename, tracks, divisions, lineage_trees):
    with open(filename, 'w') as lineage_dump:
        import pickle
        pickle.dump(tracks, lineage_dump)
        pickle.dump(divisions, lineage_dump)
        pickle.dump(lineage_trees, lineage_dump)


def analyze_lineage_dump(args):
    # load lineages
    tracks, divisions, lineage_trees = load_lineage_dump(args.lineage_dump_file)
    # find gt an proposal files that start with a number and end with h5

    # gt_filenames = [path.abspath(path.join(args.gt_path, fn))
    #                 for fn in os.listdir(args.gt_path) if fn.endswith('.h5') and fn[0].isdigit()]
    # gt_filenames.sort()
    # proposal_filenames = [path.abspath(path.join(args.proposal_path, fn))
    #                       for fn in os.listdir(args.proposal_path) if fn.endswith('.h5') and fn[0].isdigit()]
    # proposal_filenames.sort()

    # evaluate
    print("Analyzing {} lineage trees".format(len(lineage_trees)))
    taxonomies = compare_lineage_trees_to_gt(args.gt_path, args.proposal_path, lineage_trees)
    precisions = [t.precision() for t in taxonomies]
    recalls = [t.recall() for t in taxonomies]
    fmeasures = [t.f_measure() for t in taxonomies]

    def replaceNan(a):
        if np.isnan(a):
            return 0.0
        else:
            return a

    precisions = map(replaceNan, precisions)
    recalls = map(replaceNan, recalls)
    fmeasures = map(replaceNan, fmeasures)
    np.savetxt(args.out_dir.rstrip('/') + '/precisions.txt', np.array(precisions))
    np.savetxt(args.out_dir.rstrip('/') + '/recalls.txt', np.array(recalls))
    np.savetxt(args.out_dir.rstrip('/') + '/fmeasures.txt', np.array(fmeasures))
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(precisions, 100)
    plt.savefig(args.out_dir.rstrip('/') + '/precisions.pdf')

    plt.figure()
    plt.hist(recalls, 100)
    plt.savefig(args.out_dir.rstrip('/') + '/recalls.pdf')

    plt.figure()
    plt.hist(fmeasures, 100)
    plt.savefig(args.out_dir.rstrip('/') + '/fmeasures.pdf')

    if len(args.weight_filename) > 0 and os.path.isfile(args.weight_filename):
        # load weights and compute all scores
        weights = np.loadtxt(args.weight_filename)
        assert len(weights) == len(lineage_trees[0].get_feature_vector())
        scores = [np.dot(weights, lt.get_feature_vector()) for lt in lineage_trees]

        plt.figure()
        plt.scatter(scores, precisions)
        plt.savefig(args.out_dir.rstrip('/') + '/score_vs_precision.pdf')

        plt.figure()
        plt.scatter(scores, fmeasures)
        plt.savefig(args.out_dir.rstrip('/') + '/score_vs_fmeasure.pdf')


        ## -----------------------------
        ## how to test lineage tree construction:
        ## -----------------------------
        # import trackingfeatures
        # import pickle
        # import h5py
        # out_dir = '/Users/chaubold/hci/data/hufnagel2012-08-03/current-bests/m4-rerank-2015-03-24_17-29-18/'
        # ts_in = open('drosophila-7ts-2015-03-22_12-56-32.dump', 'rb')
        # ts = pickle.load(ts_in)
        # fs = pickle.load(ts_in)
        # ts.set_feature_store(fs)
        # ts_in.close()
        # def getRegionFeatures(ndim):
        # region_features = [
        #             ("RegionCenter", ndim),
        #             ("Count", 1),
        #             ("Variance", 1),
        #             ("Sum", 1),
        #             ("Mean", 1),
        #             ("RegionRadii", ndim),
        #             ("Central< PowerSum<2> >", 1),
        #             ("Central< PowerSum<3> >", 1),
        #             ("Central< PowerSum<4> >", 1),
        #             ("Kurtosis", 1),
        #             ("Maximum", 1),
        #             ("Minimum", 1),
        #             ("RegionAxes", ndim**2),
        #             ("Skewness", 1),
        #             ("Weighted<PowerSum<0> >", 1),
        #             ("Coord< Minimum >", ndim),
        #             ("Coord< Maximum >", ndim)
        #         ]
        #     return region_features
        # iteration = 0
        # track_features_filename = out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_features.h5'
        # track_features_h5 = h5py.File(track_features_filename, 'r')
        # tracks, divisions = trackingfeatures.create_and_link_tracks_and_divisions(track_features_h5, ts, getRegionFeatures(3))
        # lineage_trees = trackingfeatures.build_lineage_trees(tracks, divisions)
        # track_features_h5.close()


if __name__ == "__main__":
    """
    Executing this script will take a lineage tree dump together with ground truth
    and proposal solution h5 files (containing the events). It then evaluates the
    quality of each lineage independently.
    """

    parser = argparse.ArgumentParser(description='Evaluate precision/recall/f-measure'
                                                 'for each lineage independently')
    parser.add_argument('--lineage-dump', required=True, type=str, dest='lineage_dump_file',
                        help='Filename of the lineage dump created by a multitrack run')
    parser.add_argument('--gt-path', required=True, type=str, dest='gt_path',
                        help='Path to folder containing the ground truth .h5 files')
    parser.add_argument('--proposal-path', required=True, type=str, dest='proposal_path',
                        help='Path to folder containing the proposal .h5 files')
    parser.add_argument('--out', type=str, dest='out_dir', default='.', help='Output directory')
    parser.add_argument('--weights', type=str, dest='weight_filename', default='',
                        help='If a filename for the weights is specified, this tool also plots scores vs precision etc')

    args = parser.parse_args()

    analyze_lineage_dump(args)

