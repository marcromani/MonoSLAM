#ifndef MAP_H
#define MAP_H

#include <vector>

#include "opencv2/core/core.hpp"

#include "camera.hpp"
#include "feature.hpp"

class Map {

public:

    Camera camera;                      // Camera

    std::vector<Feature> features;      // Initialized map features (used for tracking)
    std::vector<Feature> candidates;    // Pre-initialized features (updated by particle filter)

    int patchSize;                      // Size of the feature planar patch, in pixels

    int minFeatureDensity;              // Minimum number of features per frame
    int maxFeatureDensity;              // Maximum number of features per frame

    double failTolerance;               // Maximum ratio of matching failures before feature rejection

    int numVisibleFeatures;             // Number of currently visible features (a subset of those in sight)
    std::vector<cv::Point2i> inview;    // Image positions of currently in sight features

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
        numVisibleFeatures(0),
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
        numVisibleFeatures(0),
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

    /*
     * Updates the camera and features data with the map state vector and map state
     * covariance matrix data. Only shallow copies are performed. This function should
     * be called whenever the current state of the system must be broadcast to each
     * of its parts. This method exists because the map state (x and P) is continuously
     * updated by the EKF and these changes will not be directly reflected in the camera
     * and features instances.
     */
    void broadcastData();

    /*
     * Detects corners and initializes new feature candidates whenever there are not
     * enough visible features and no candidates are being tracked. Returns a boolean
     * indicating whether new candidates were detected or not.
     *
     * frame    Grayscale frame
     */
    bool trackNewCandidates(const cv::Mat& frame);

    /*
     * Draws features that are (theoretically) in view.
     *
     * frame    Camera frame
     */
    void drawInViewFeatures(const cv::Mat& frame);

private:

    /*
     * Adds an initial feature to the map. Its position is known with zero uncertainty
     * hence the associated covariance matrix is the zero matrix. Moreover, since the
     * feature patch lies on the chessboard pattern its normal vector is taken to be
     * the z axis unit vector. The feature is also set as visible. If the feature patch
     * exceeds the frame boundaries the feature is not initialized and false is returned.
     *
     * frame    Grayscale frame
     * pos2D    Feature position in the image, in pixels
     * pos3D    Feature position, in world coordinates
     * R        Camera rotation (from world to camera)
     * t        Camera position, in world coordinates
     */
    bool addInitialFeature(const cv::Mat& frame, const cv::Point2f& pos2D, const cv::Point3f& pos3D,
                           const cv::Mat& R, const cv::Mat& t);

    /*
     * Sets the map as it was upon creation.
     */
    void reset();

    /*
     * Returns corners found by the Shi-Tomasi corner detector. In particular, the
     * number of computed corners is such that when the associated pre-initialized
     * features are added to the map for tracking, the number of visible features
     * in the current frame will (hopefully) be the maximum allowed.
     *
     * frame    Grayscale frame
     */
    cv::Mat findCorners(const cv::Mat& frame);
};

#endif