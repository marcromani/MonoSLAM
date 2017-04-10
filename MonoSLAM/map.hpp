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

    int minFeatureDensity;              // Minimum number of features per frame
    int maxFeatureDensity;              // Maximum number of features per frame

    double failTolerance;               // Max ratio of matching failures before a feature is rejected

    cv::Mat x;                          // Map state (camera and features states)
    cv::Mat P;                          // Map state covariance matrix

    /*
     * Constructor for a lens distortion camera model.
     */
    Map(const cv::Mat& K,
        const cv::Mat& distCoeffs,
        const cv::Size& frameSize,
        int patchSize_,
        int minFeatureDensity_,
        int maxFeatureDensity_,
        double failTolerance_) :
        camera(K, distCoeffs, frameSize),
        patchSize(patchSize_),
        minFeatureDensity(minFeatureDensity_),
        maxFeatureDensity(maxFeatureDensity_),
        failTolerance(failTolerance_),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {}

    /*
     * Constructor for a pinhole camera model.
     */
    Map(const cv::Mat& K,
        const cv::Size& frameSize,
        int patchSize_,
        int minFeatureDensity_,
        int maxFeatureDensity_,
        double failTolerance_) :
        camera(K, frameSize),
        patchSize(patchSize_),
        minFeatureDensity(minFeatureDensity_),
        maxFeatureDensity(maxFeatureDensity_),
        failTolerance(failTolerance_),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {}

    /*
     * Initializes the map by detecting a known chessboard pattern. It provides an
     * absolute scale reference, some manually initialized features to track, and
     * the world frame origin and initial camera pose. The world frame origin is
     * positioned at the top left corner of the pattern, with the x axis pointing
     * along the pattern height and the y axis pointing along the pattern width.
     * Note that the z axis is determined by the right-hand rule (see solvePnP).
     * It is assumed the camera has zero linear and angular velocities.
     *
     * frame            Grayscale frame
     * patternSize      Pattern dimensions (see findChessboardCorners)
     * squareSize       Size of a chessboard square, in meters
     * var              Variances of the initial camera state components
     */
    bool initMap(const cv::Mat& frame, const cv::Size& patternSize, double squareSize,
                 const std::vector<double>& var);

    bool trackNewCandidates(const cv::Mat& frame);

private:

    /*
     * Adds an initial feature to the map. Its position is known with zero uncertainty
     * hence the associated covariance matrix is the zero matrix. Moreover, since the
     * feature patch lies on the chessboard pattern its normal vector is taken to be
     * the z axis unit vector.
     *
     * frame    Grayscale frame
     * pos2D    Feature position in the image, in pixels
     * pos3D    Feature position, in world coordinates
     * R        Camera rotation (world to camera)
     * t        Camera position, in world coordinates
     */
    void addInitialFeature(const cv::Mat& frame, const cv::Point2f& pos2D, const cv::Point3f& pos3D,
                           const cv::Mat& R, const cv::Mat& t);

    /*
     * Updates the camera and features data with the map state vector and covariance matrix
     * data. Only shallow copies are performed. This function should be called whenever the
     * current state of some (or all) of the system parts is required.
     */
    void update();

    vector<Point2f> findCorners(const Mat& frame);
};

#endif