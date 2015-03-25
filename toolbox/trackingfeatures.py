import h5py
import pgmlink as track
import numpy as np
import math

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

    weight_to_track_feature_map = {
         0: 'sq_diff_RegionCenter',
         1: 'sq_diff_Count',
         2: 'sq_diff_Mean',
         3: 'sq_diff_Variance',
         4: 'sq_accel_RegionCenter',
         5: 'sq_accel_Count',
         6: 'sq_accel_Mean',
         7: 'sq_accel_Variance',
         8: 'angles_RegionCenter',
         9: 'outlier_id_RegionCenter',
        10: 'outlier_id_Count',
        11: 'outlier_id_Mean',
        12: 'outlier_id_Variance',
        13: 'diff_outlier_RegionCenter',
        14: 'diff_outlier_Count',
        15: 'diff_outlier_Mean',
        16: 'diff_outlier_Variance',
        17: 'Count',
        18: 'Mean',
        19: 'Variance',
        20: 'Sum',
        21: 'Kurtosis',
        22: 'Skewness',
        23: 'Weighted<PowerSum<0> >'
    }

    weight_to_division_feature_map = {
         0: 'child_decel_Count',
         1: 'child_decel_Count_outlier_score',
         2: 'child_decel_Mean',
         3: 'child_decel_Mean_outlier_score',
         4: 'child_decel_RegionCenter',
         5: 'child_decel_RegionCenter_outlier_score',
         6: 'child_decel_Variance',
         7: 'child_decel_Variance_outlier_score',
         8: 'sq_diff_Count',
         9: 'sq_diff_Count_outlier_score',
        10: 'sq_diff_Mean',
        11: 'sq_diff_Mean_outlier_score',
        12: 'sq_diff_RegionCenter',
        13: 'sq_diff_RegionCenter_outlier_score',
        14: 'sq_diff_Variance',
        15: 'sq_diff_Variance_outlier_score'
    }

    def get_feature_vector_size(self):
        return 2 * (len(self.weight_to_division_feature_map) + len(self.weight_to_track_feature_map))

    def get_feature_vector(self):
        """
        This should return the feature vector such that it can be multiplied by the weight vector to return this track's score
        @returns a numpy array
        """
        raise NotImplementedError("Please Specialize this method")

    def extract(self, track_features_h5):
        """
        Extract all important information about this part of the lineage from the features h5 file
        """
        raise NotImplementedError("Please Specialize this method")

    def compute_score(self, weights):
        """
        Returns the score as linear combination of feature vector and supplied weights
        """
        feature_vec = self.get_feature_vector()
        return np.dot(weights, feature_vec)


class Track(LineagePart):
    def __init__(self, track_id):
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

        # set of features of this track
        self.features = {}

    def extract_region_features(self, track_features_h5, ts, region_features):
        """
        Extract the region features of this track's traxels from the traxelstore,
        and compute mean an variance.
        TODO: mean + variance of some features doesn't make sense at all! (e.g. radii)
        """
        assert(self.traxels != None and len(self.traxels) > 0)
        
        # temporary storage for features over traxels
        f = {}

        for timestep, idx in self.traxels.transpose():
            traxel = ts.get_traxel(int(idx), int(timestep))
            for feat_name, feat_dims in region_features:
                if not feat_name in f:
                    f[feat_name] = []
                f[feat_name].append(get_feature_vector(traxel, feat_name, feat_dims))

        # create a feature matrix per feature, and store mean + max
        for k, v in f.iteritems():
            feature_matrix = np.array(v)
            # compute mean & variance
            self.features['mean_' + k] = np.mean(feature_matrix)
            self.features['var_'  + k] = np.var(feature_matrix)

    def extract(self, track_features_h5):
        """
        Get the higher order features of this track from the HDF5 file,
        and compute the mean of each feature.
        """
        t_features = track_features_h5['tracks/' + str(self.track_id) + '/']
        self.length = int(t_features['track_length'].value[0,0])
        self.traxels = t_features['traxels'].value
        assert(self.traxels.shape[1] > 0)
        self.start_traxel_id = tuple(self.traxels[:,0])
        self.end_traxel_id = tuple(self.traxels[:,-1])

        # get all features from the HDF5 file
        for v in LineagePart.weight_to_track_feature_map.values():
            try:
                feature_matrix = t_features[v].value
                self.features['mean_' + v] = np.mean(feature_matrix)
                self.features['var_'  + v] = np.var(feature_matrix)
            except:
                #print("Could not find feature {} for track {}".format(v, self.track_id))
                pass


    def get_feature_vector(self):
        """
        The feature vector is long enough to store all features, but here we
        only fill in the track features. Each feature name is used twice,
        for mean and variance of the feature.
        """
        result = np.zeros(self.get_feature_vector_size())

        for k, v in LineagePart.weight_to_track_feature_map.iteritems():
            try:
                result[2 * k] = self.features['mean_' + v]
                result[2 * k + 1] = self.features['var_' + v]
            except:
                pass

        return result

    def get_expanded_feature_vector(self, series_expansion_range):
        fv = self.get_feature_vector()
        # TODO: build properly sized numpy array!
        resultFV = []
        for i in range(series_expansion_range[0], series_expansion_range[1]):
            resultFV += fv * math.pow(self.length, i)
        return resultFV


class Division(LineagePart):
    def __init__(self, division_id):
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
        self.parent_traxel_id = tuple(traxels[:,0])
        self.children_traxel_ids = [tuple(traxels[:,1]), tuple(traxels[:,2])]

        # get all features from the HDF5 file
        for v in LineagePart.weight_to_division_feature_map.values():
            try:
                feature_matrix = d_features[v].value
                self.features['mean_' + v] = np.mean(feature_matrix)
                self.features['var_'  + v] = np.var(feature_matrix)
            except:
                # print("Could not find feature {} for division {}".format(v, self.division_id))
                pass

    def get_feature_vector(self):
        """
        The feature vector is long enough to store all features, but here we
        only fill in the division features. Each feature name is used twice,
        for mean and variance of the feature.
        """
        offset = 2 * len(LineagePart.weight_to_track_feature_map)
        result = np.zeros(self.get_feature_vector_size())

        for k, v in LineagePart.weight_to_division_feature_map.iteritems():
            try:
                result[offset + 2 * k] = self.features['mean_' + v]
                result[offset + 2 * k + 1] = self.features['var_' + v]
            except:
                pass

        return result


class LineageTree(LineagePart):
    def __init__(self, lineage_tree_id, track, tracks, divisions):
        self.lineage_tree_id = lineage_tree_id
        self.tracks = [track]
        self.divisions = []
        self.length = 0

        # follow the supplied track along divisions
        while self.tracks[-1].end_division_id != -1:
            self.divisions.append(divisions[self.tracks[-1].end_division_id])
            self.tracks.append(tracks[self.divisions[-1].children_track_ids[0]])

    def get_feature_vector(self):
        # TODO: weight according to num tracks and divisions
        result = np.zeros(self.get_feature_vector_size())
        
        for t in self.tracks:
            result += t.get_feature_vector()

        for d in self.divisions:
            result += d.get_feature_vector()

        return result

    def get_expanded_feature_vector(self, series_expansion_range):
        result = []
        for t in self.tracks:
            result += t.get_expanded_feature_vector(series_expansion_range)

        for d in self.divisions:
            result += d.get_feature_vector()

        return result


def create_and_link_tracks_and_divisions(track_features_h5, ts, region_features):
    # storage for all tracks and divisions
    tracks = {}
    divisions = {}

    # mapping of traxelID at front and back of track to track_id
    track_starts_with_traxel_id = {}
    track_ends_with_traxel_id = {}

    for track_id in track_features_h5['tracks'].keys():
        track_id_int = int(track_id)
        t = Track(track_id_int)
        t.extract(track_features_h5)
        t.extract_region_features(track_features_h5, ts, region_features)

        # store in container
        tracks[track_id_int] = t
        
        # create mappings
        track_starts_with_traxel_id[t.start_traxel_id] = track_id_int
        track_ends_with_traxel_id[t.end_traxel_id] = track_id_int

    for division_id in track_features_h5['divisions'].keys():
        division_id_int = int(division_id)
        d = Division(division_id_int)
        d.extract(track_features_h5)

        # find the tracks that are connected by this division
        d.parent_track_id = track_ends_with_traxel_id[d.parent_traxel_id]
        d.children_track_ids[0] = track_starts_with_traxel_id[d.children_traxel_ids[0]]
        d.children_track_ids[1] = track_starts_with_traxel_id[d.children_traxel_ids[1]]

        # update information in the tracks
        tracks[d.parent_track_id].end_division_id = division_id_int
        tracks[d.children_track_ids[0]].start_division_id = division_id_int
        tracks[d.children_track_ids[1]].start_division_id = division_id_int

        # store in container
        divisions[division_id_int] = d
    return tracks, divisions

def build_lineage_trees(tracks, divisions):
    # select all tracks that have no parent division (appearances)
    appearances = [t for t in tracks.values() if t.start_division_id == -1]

    # create a new lineage tree for each of those
    lineageTrees = [LineageTree(l_id, t, tracks, divisions) for l_id, t in enumerate(appearances)]

    return lineageTrees

def extract_features_and_compute_score(reranker_weight_filename, 
                                        outlier_svm_filename, 
                                        out_dir, 
                                        iteration, 
                                        hypotheses_graph, 
                                        ts, 
                                        fov, 
                                        feature_vector_filename, 
                                        region_features):
    
    # extract higher order features and per-track features
    print("Extracting features of solution {}".format(iteration))
    track_features_filename = out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_features.h5'
    feature_extractor = track.TrackingFeatureExtractor(hypotheses_graph, fov)
    overall_score = 0
    
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
    feature_extractor.set_track_feature_output_file(track_features_filename)
    feature_extractor.compute_features()

    # create complete lineage trees with extracted features:
    with h5py.File(track_features_filename, 'r') as track_features_h5:
        tracks, divisions = create_and_link_tracks_and_divisions(track_features_h5, ts, region_features)
        lineage_trees = build_lineage_trees(tracks, divisions)

    # accumulate feature vectors
    feature_vector = lineage_trees[0].get_feature_vector()
    for lt in lineage_trees[1:]:
        feature_vector += lt.get_feature_vector()

    # if reranker weights are already given, compute overall, and track scores and plot a histogram
    if reranker_weight_filename and len(reranker_weight_filename) > 0:
        reranker_weights = np.loadtxt(reranker_weight_filename)
        overall_score = np.dot(reranker_weights, feature_vector)

        track_scores = [t.compute_score(reranker_weights) for t in tracks.values()]
        division_scores = [d.compute_score(reranker_weights) for d in divisions.values()]
        lineage_scores = [l.compute_score(reranker_weights) for l in lineage_trees]

        for s_name, scores in [("track_scores", track_scores), 
                               ("division_scores", division_scores), 
                               ("lineage_scores", lineage_scores)]:
            print("Saving {} for iteration {}".format(s_name, iteration))
            np.savetxt(out_dir.rstrip('/') + '/iter_' + str(iteration) + '/' + s_name + '.txt', scores)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(scores, 100)
            plt.savefig(out_dir.rstrip('/') + '/iter_' + str(iteration) + '/' + s_name + '.pdf')


        print("Overall score of tracking is: {}".format(overall_score))

    return feature_vector, overall_score, ["feature_names"]

## -----------------------------
## how to test:
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
#     region_features = [
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