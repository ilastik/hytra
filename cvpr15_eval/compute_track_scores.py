#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import h5py
import argparse

def get_track_feature_matrix(track_features_filename, region_features, normalize=False):
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
                print("Warning: could not extract some feature of track {}, setting it's feature value to zero".format(track_id))

            # divide by num traxels to get average
            if normalize:
                merged_features = [f / len(traxels) for f in merged_features]

            track_feature_matrix = np.vstack([track_feature_matrix, merged_features])

    # create list of names of the features, to ease later analysis and plotting
    track_feature_names = []
    for f, dim in region_features:
        track_feature_names += [f + '[' + str(i) + ']' for i in range(dim)]

    return track_feature_matrix, track_feature_names


def getRegionFeatures(ndim):
    region_features = [
            ("RegionCenter", ndim),
            ("Count", 1),
            ("Variance", 1),
            ("Sum", 1),
            ("Mean", 1),
            ("RegionRadii", ndim),
            ("Central< PowerSum<2> >", 1),
            ("Central< PowerSum<3> >", 1),
            ("Central< PowerSum<4> >", 1),
            ("Kurtosis", 1),
            ("Maximum", 1),
            ("Minimum", 1),
            ("RegionAxes", ndim**2),
            ("Skewness", 1),
            ("Weighted<PowerSum<0> >", 1),
            ("Coord< Minimum >", ndim),
            ("Coord< Maximum >", ndim)
        ]
    return region_features


def extract_features_and_compute_score(options):
    # if reranker weights are already given, compute overall, and track scores
    if options.reranker_weight_file and len(options.reranker_weight_file) > 0:
        reranker_weights = np.loadtxt(options.reranker_weight_file)

        # track scores
        track_feature_matrix = get_track_feature_matrix(options.track_features, getRegionFeatures(options.ndim), True)
        with h5py.File(options.track_features, 'r') as track_features:
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

                # if len(options.load_outlier_svm) > 0:
                #     outlier_svm_score = track_features['track_outliers_svm'][int(track_id)]
                #     score += outlier_svm_score * reranker_weights[70]
                #     # outlier_svm_score = track_features['track_outliers_svm'][int(track_id)]
                #     # score += outlier_svm_score * reranker_weights[114]
                track_length = np.array(track_features['tracks/' + track_id + '/track_length'])
                track_scores.append([float(track_id), score, int(track_length[0])])

            #Todo: analyze all divisions

            #Todo: stitch tracks (match last track traxel id to first division traxel id and vice versa)
            #       and then compute quality measures
            track_scores = np.array(track_scores)
            print("Saving track scores")
            np.savetxt(options.out_dir.rstrip('/') + '/track_scores.txt', track_scores)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(track_scores[:, 1], 100)
            plt.savefig(options.out_dir.rstrip('/') + '/track_scores.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute Track scores of proposal')

    # file paths
    parser.add_argument('--track-features', type=str, dest='track_features', required=True,
                        help='Proposal labeling files')
    parser.add_argument('--reranker-weights', type=str, dest='reranker_weight_file', required=True,
                        help='file containing the learned reranker weights')
    parser.add_argument('--num-dims', type=int, dest='ndim', required=True,
                        help='number of dimensions (2 or 3)')
    parser.add_argument('-o', required=True, type=str, dest='out_dir',
                        help='Directory where to save results')

    options = parser.parse_args()

    extract_features_and_compute_score(options)