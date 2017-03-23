#ifndef MAP_H
#define MAP_H

#include <vector>

#include "opencv2/core/core.hpp"

#include "camera.hpp"
#include "feature.hpp"

class Map {

public:

    Camera camera;                      // Camera
    std::vector<Feature> features;      // Initialized map features
    std::vector<Feature> candidates;    // Pre-initialized features

    int patchSize;                      // Size of the feature planar patch

    cv::Mat x;                          // Map state (camera and features states)
    cv::Mat P;                          // Map state covariance matrix

    /*
     * Constructor for a lens distortion camera model.
     */
    Map(const cv::Mat& K,
        const cv::Mat& distCoeffs,
        const cv::Size& frameSize,
        int patchSize_) :
        camera(K, distCoeffs, frameSize),
        patchSize(patchSize_),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {}

    /*
     * Constructor for a pinhole camera model.
     */
    Map(const cv::Mat& K,
        const cv::Size& frameSize,
        int patchSize_) :
        camera(K, frameSize),
        patchSize(patchSize_),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {}

    /*
     * Initializes the map by detecting a known chessboard pattern. It provides an
     * absolute scale reference, some manually initialized features to track, and
     * the world frame origin and initial camera pose.
     *
     * frame            Camera frame
     * patternSize      Pattern dimensions (see findChessboardCorners)
     * squareSize       Size of a chessboard square, in meters
     * var              Variance of the initial camera state
     */
    bool initMap(const cv::Mat& frame, const cv::Size& patternSize, double squareSize,
                 const std::vector<double>& var);

private:

    void addInitialFeature(const cv::Mat& frame, const cv::Point2f& pos2D, const cv::Point3f& pos3D,
                           const cv::Mat& cov, const cv::Mat& R, const cv::Mat& t);

    void update();
};

#endif