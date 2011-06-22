#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>

#include "vigra/multi_array.hxx"
#include "vigra/matrix.hxx"
#include "vigra/eigensystem.hxx"
#include "vigra/labelvolume.hxx"

#include "ConfigFeatures.hxx"
#include "SGFeatures.hxx"


namespace features {


    //constants for selecting the features
    typedef unsigned int feature_flag;
    const feature_flag FT_NONE 					= 0x00000000;
    const feature_flag FT_VOLUME 				= 0x00000001;
    const feature_flag FT_BOUNDING_BOX			= 0x00000002;
    const feature_flag FT_POSITION				= 0x00000100;
    const feature_flag FT_CENTER_OF_MASS		= 0x00000200;
    const feature_flag FT_PRINCIPAL_COMPONENTS	= 0x00000400;
    const feature_flag FT_INTENSITY				= 0x00010000;
    const feature_flag FT_INTENSITY_MIN_MAX		= 0x00020000;
    const feature_flag FT_EXPERIMENTAL_PAIR     = 0x00100000;
    const feature_flag FT_EXPERIMENTAL_SGF      = 0x00200000;
    const feature_flag FT_ALL					= 0xffffffff;




    /** Supervoxel volume feature
      * Count the number of voxels the supervoxel consists of.
      * Output stucture:
      * size = 1
      * [0] volume
      */
    feature_array extractVolume(three_set& coordinates, feature_array& intensities);


    /** (Unweighted) Mean position feature
      * Calculate the mean position and its higher central moments
      * Output stucture:
      * size = 12
      * [0..2] mean x,y,z coordinates
      * [3..5] variance of x,y,z coordinates
      * [6..8] skew of x,y,z coordinates
      * [9..11] kurtosis of x,y,z coordinates
      */
    feature_array extractPosition(three_set& coordinates, feature_array& intensities);


    /** Weighted Mean position feature
      * Calculate the intensity weighted mean position and its higher central moments
      * Output stucture:
      * size = 12
      * [0..2] weighted mean x,y,z coordinates
      * [3..5] variance of x,y,z coordinates
      * [6..8] skew of x,y,z coordinates
      * [9..11] kurtosis of x,y,z coordinates
      */
    feature_array extractWeightedPosition(three_set& coordinates, feature_array& intensities);


    /** Principal components feature
      * Calculate the principal components of the voxel distribution
      * Output stucture:
      * size = 12
      * [0..2] Eigenvalues of covariance matrix
      * [3..5] Eigenvector of eigenvalue [0]
      * [6..8] Eigenvector of eigenvalue [1]
      * [9..11] Eigenvector of eigenvalue [2]
      */
    feature_array extractPrincipalComponents(three_set& coordinates, feature_array& intensities);


    /** Bounding Box feature
      * Find the smallest possible box that contains the whole supervoxel
      * Output stucture:
      * size = 7
      * [0] Lower x position
      * [1] Lower y position
      * [2] Lower z position
      * [3] Upper x position
      * [4] Upper y position
      * [5] Upper z position
      * [6] Fill factor: <supervoxel volume> / <size of bounding box>
      */
    feature_array extractBoundingBox(three_set& coordinates, feature_array& intensities);


    /** Intensity feature
      * Calculate the mean intensity and its central moments of all super voxels
      * Output stucture:
      * size = 4
      * [0] Mean of intensity distribution
      * [1] Variance of intensity distribution
      * [2] Skew of Intensity distribution
      * [3] Kurtosis of Intensity distribution
      */
    feature_array extractIntensity(three_set& coordinates, feature_array& intensities);


    /** Minimum/Maximum Intensity feature
      * Find the minimum and the maximum intensity of a super voxel and find the
      * quantiles of the intensity distribution.
      * Output stucture:
      * size = 9
      * [0] Minimum intensity
      * [1] Maximum intensity
      * [2] 5% quantile
      * [3] 10% quantile
      * [4] 20% quantile
      * [5] 50% quantile
      * [6] 80% quantile
      * [7] 90% quantile
      * [8] 95% quantile
      */
    feature_array extractMinMaxIntensity(three_set& coordinates, feature_array& intensities);

    /** Minimum/Maximum Intensity feature
      * Find the minimum and the maximum intensity of a super voxel and find the
      * quantiles of the intensity distribution.
      * Output stucture:
      * size = 4
      * [0] Maximum intensity
      * [1] x1 coord
      * [2] x2 coord
      * [3] x3 coord
      */
    feature_array extractMaxIntensity(three_set& coordinates, feature_array& intensities);



    /** Pairwise features
      * Calculate average values of differences of neighboring intensity values.
      * Attention: Features are not fixed yet.
      * Output stucture:
      * size = 4
      * [0] Average sum over absolute distances
      * [1] Average sum over squared distances
      * [2] Average symmetric first derivative
      * [3] Average second derivative
      */
    feature_array extractPairwise(three_set &coordinates, feature_array &intensities);


    /** Histogram of Oriented Gradients
      * Don't use them. They don't improve anything, just cost computation time.
      * RF variable importance was about 0.1 of informative features
      * Output stucture:
      * size = 20
      */
    feature_array extractHOG(three_set &coordinates, feature_array &intensities);


    /** Experimental SGF Features
      *
      * Output stucture:
      * size = 48
      */
    feature_array extractSGF(three_set &coordinates, feature_array &intensities);


} /* namespace features */
