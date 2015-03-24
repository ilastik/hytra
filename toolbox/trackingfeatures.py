import h5py
import pgmlink as track
import numpy as np

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


def construct_feature_vector(feature_matrix_basename,
                             proposal_basename,
                             iteration,
                             feature_vector):
    feature_matrix = np.loadtxt(feature_matrix_basename + '_{}.txt'.format(iteration))
    proposal_label = np.loadtxt(proposal_basename + '_{}.txt'.format(iteration))

    new_features = np.dot(proposal_label, feature_matrix)
    feature_vector = list(feature_vector) + list(new_features)
    return feature_vector


def get_track_feature_matrix(ts, track_features_filename, region_features, normalize=False):
    num_features = sum([i for n,i in region_features])
    track_feature_matrix = np.array([]).reshape(0, num_features)
    with h5py.File(track_features_filename, 'r') as track_features:
        # examine all tracks
        for track_id in track_features['tracks'].keys():
            merged_features = np.zeros(num_features)

            try:
                # get accumulated feature vector
                traxels = np.array(track_features['tracks/' + track_id + '/traxels']).transpose()
                for timestep, idx in traxels:
                    feats = []
                    traxel = ts.get_traxel(int(idx), int(timestep))
                    for feat_name, feat_dims in region_features:
                        feats += get_feature_vector(traxel, feat_name, feat_dims)

                    merged_features += np.array(feats)
            except:
                print("Warning: could not extract some feature of track {}, setting it's feature value to zero")

            # divide by num traxels to get average
            if normalize:
                merged_features = [f / len(traxels) for f in merged_features]

            track_feature_matrix = np.vstack([track_feature_matrix, merged_features])

    # create list of names of the features, to ease later analysis and plotting
    track_feature_names = []
    for f, dim in region_features:
        track_feature_names += [f + '[' + str(i) + ']' for i in range(dim)]

    return track_feature_matrix, track_feature_names


def extract_features_and_compute_score(reranker_weight_filename, 
                                        outlier_svm_filename, 
                                        out_dir, 
                                        iteration, 
                                        hypotheses_graph, 
                                        ts, 
                                        fov, 
                                        feature_vector_filename, 
                                        region_features):
    import pickle
    # extract higher order features and per-track features
    print("Extracting features of solution {}".format(iteration))
    track_features_filename = out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_features.h5'
    feature_extractor = track.TrackingFeatureExtractor(hypotheses_graph, fov)
    if len(outlier_svm_filename) > 0:  # when the svm was trained in this run, it automatically sets load_outlier_svm
        with open(outlier_svm_filename, 'r') as svm_dump:
            outlier_track_svm = pickle.load(svm_dump)
            outlier_division_svm = pickle.load(svm_dump)

            feature_extractor.set_track_svm(outlier_track_svm)
            feature_extractor.set_division_svm(outlier_division_svm)
            print("Trained outlier SVM loaded from " + outlier_svm_filename)
    feature_extractor.set_track_feature_output_file(track_features_filename)
    feature_extractor.compute_features()
    feature_vector = [f for f in feature_extractor.get_feature_vector()]
    feature_names = [feature_extractor.get_feature_description(f) for f in range(len(feature_vector))]

    track_feature_matrix, track_feature_names = get_track_feature_matrix(ts, track_features_filename, region_features, False)
    sum_of_track_features = np.sum(track_feature_matrix, axis=0)

    feature_vector += list(sum_of_track_features)
    feature_names += track_feature_names
    overall_score = 0

    # if reranker weights are already given, compute overall, and track scores
    if reranker_weight_filename and len(reranker_weight_filename) > 0:
        reranker_weights = np.loadtxt(reranker_weight_filename)
        overall_score = np.dot(reranker_weights, np.array(feature_vector))

        # track scores
        track_feature_matrix = get_track_feature_matrix(ts, track_features_filename, region_features, True)
        with h5py.File(track_features_filename, 'r') as track_features:
            track_scores = []
            # examine all tracks, extract HO features for quality measures
            for track_id in track_features['tracks'].keys():

                # ugly way of using the learned weights to compute track quality measurements
                score = 0

                def get_feature_score(mean_weight_idx, name):
                    try:
                        value = np.array(track_features['tracks/' + track_id + '/' + name])
                        return reranker_weights[mean_weight_idx] * np.mean(value) \
                               + reranker_weights[mean_weight_idx + 1] * np.var(value)
                    except:
                        print("Track {} has no feature: {}".format(track_id, name))
                        return 0

                score += get_feature_score(0, 'sq_diff_RegionCenter')
                score += get_feature_score(4, 'sq_diff_Count')
                score += get_feature_score(8, 'sq_diff_Mean')
                score += get_feature_score(12, 'sq_diff_Variance')
                score += get_feature_score(16, 'sq_accel_RegionCenter')
                score += get_feature_score(20, 'sq_accel_Count')
                score += get_feature_score(24, 'sq_accel_Mean')
                score += get_feature_score(28, 'sq_accel_Variance')
                score += get_feature_score(32, 'angles_RegionCenter')
                score += get_feature_score(38, 'outlier_id_RegionCenter')
                score += get_feature_score(42, 'outlier_id_Count')
                score += get_feature_score(46, 'outlier_id_Mean')
                score += get_feature_score(50, 'outlier_id_Variance')
                score += get_feature_score(54, 'diff_outlier_RegionCenter')
                score += get_feature_score(58, 'diff_outlier_Count')
                score += get_feature_score(62, 'diff_outlier_Mean')
                score += get_feature_score(66, 'diff_outlier_Variance')

                if len(outlier_svm_filename) > 0:
                    outlier_svm_score = track_features['track_outliers_svm'][int(track_id)]
                    score += outlier_svm_score * reranker_weights[70]
                    # outlier_svm_score = track_features['track_outliers_svm'][int(track_id)]
                    # score += outlier_svm_score * reranker_weights[114]

                track_scores.append([float(track_id), score])

            #Todo: analyze all divisions

            #Todo: stitch tracks (match last track traxel id to first division traxel id and vice versa)
            #       and then compute quality measures
            track_scores = np.array(track_scores)
            print("Saving track scores for iteration " + str(iteration))
            np.savetxt(out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_scores.txt', track_scores)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(track_scores[:, 1], 100)
            plt.savefig(out_dir.rstrip('/') + '/iter_' + str(iteration) + '/track_scores.pdf')

        print("Overall score of tracking is: {}".format(score)) # bullshit, is set to zero for each track!

    return feature_vector, overall_score, feature_names