#ifndef MAP_H
#define MAP_H

#include <vector>

#include "opencv2/core/core.hpp"

#include "camera.hpp"
#include "feature.hpp"

class Map {

public:

    Camera camera;                          // Camera

    std::vector<Feature> features;          // Initialized map features (used for tracking)
    std::vector<Feature> candidates;        // Pre-initialized features (updated by particle filter)

    int patchSize;                          // Size of the feature planar patch, in pixels

    int minFeatureDensity;                  // Minimum number of features per frame
    int maxFeatureDensity;                  // Maximum number of features per frame

    double failTolerance;                   // Maximum ratio of matching failures before feature rejection

    int numVisibleFeatures;                 // Number of currently visible features (a subset of those in sight)
    std::vector<cv::Point2d> inviewPos;     // Image positions of currently (predicted) in sight features

    cv::Mat x;                              // Map state (camera and features states)
    cv::Mat P;                              // Map state covariance matrix

    cv::Mat A;                              // Linear and angular accelerations noise covariance matrix
    cv::Mat R;                              // Measurement noise variances, in pixels

    /*
     * Constructor for a lens distortion camera model.
     */
    Map(const cv::Mat& K,
        const cv::Mat& distCoeffs,
        const cv::Size& frameSize,
        int patchSize_,
        int minFeatureDensity_,
        int maxFeatureDensity_,
        double failTolerance_,
        const cv::Mat& accelerationVariances,
        const cv::Mat& measurementNoiseVariances) :
        camera(K, distCoeffs, frameSize),
        patchSize(patchSize_),
        minFeatureDensity(minFeatureDensity_),
        maxFeatureDensity(maxFeatureDensity_),
        failTolerance(failTolerance_),
        numVisibleFeatures(0),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {

        A = cv::Mat::diag(accelerationVariances);
        measurementNoiseVariances.copyTo(R);
    }

    /*
     * Constructor for a pinhole camera model.
     */
    Map(const cv::Mat& K,
        const cv::Size& frameSize,
        int patchSize_,
        int minFeatureDensity_,
        int maxFeatureDensity_,
        double failTolerance_,
        const cv::Mat& accelerationVariances,
        const cv::Mat& measurementNoiseVariances) :
        camera(K, frameSize),
        patchSize(patchSize_),
        minFeatureDensity(minFeatureDensity_),
        maxFeatureDensity(maxFeatureDensity_),
        failTolerance(failTolerance_),
        numVisibleFeatures(0),
        x(13, 1, CV_64FC1, cv::Scalar(0)),
        P(13, 13, CV_64FC1, cv::Scalar(0)) {

        A = cv::Mat::diag(accelerationVariances);
        measurementNoiseVariances.copyTo(R);
    }

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
     * Applies the first step of the Extended Kalman Filter. In particular, given the
     * current state estimate x_k|k and the current state covariance matrix P_k|k, it
     * computes the next a priori state estimate x_k+1|k (via the camera motion model)
     * and its associated covariance matrix P_k+1|k.
     *
     * dt    Time interval between the current and past frames
     */
    void predict(double dt);

    /*
     * Applies the update step of the Extended Kalman Filter. In particular, given the
     * predicted (a priori) state estimate x_k+1|k and the associated covariance matrix
     * P_k+1|k, it computes the corrected (a posteriori) estimate x_k+1|k+1 and its
     * covariance P_k+1|k+1 by measuring known, currently visible map features.
     *
     * gray     Grayscale frame, used for feature matching
     * frame    Color frame, used as a canvas to draw features on
     */
    void update(const cv::Mat& gray, cv::Mat& frame);

    void updateCandidates(const cv::Mat& gray, cv::Mat& frame);

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

    /*
     * Applies the state transition function to the current state estimate. It is mandatory
     * that this method be called after computeProcessMatrix and computeProcessNoiseMatrix
     * since the latter expect the last a posteriori (corrected) state estimate in the state
     * vector x.
     *
     * dt    Time interval between the current and past frames
     */
    void applyMotionModel(double dt);

    /*
     * Returns the Jacobian matrix of the state transition function with respect to the
     * complete state (r, q, v, w, f1, ..., fn). This matrix is evaluated at the current
     * state estimate on the assumption that there is zero process noise. This method must
     * be called before applyMotionModel since the latter updates the current estimate x.
     *
     * dt    Time interval between the current and past frames
     */
    cv::Mat computeProcessMatrix(double dt);

    /*
     * Returns the product W * Q * W^t, where Q is the process noise covariance matrix
     * and W is the Jacobian matrix of the state transition function with respect to
     * the noise vector (V1, V2, V3, W1, W2, W3). The latter is evaluated at the current
     * state estimate on the assumption that there is zero process noise. This method must
     * be called before applyMotionModel since the latter updates the current estimate x.
     *
     * dt    Time interval between the current and past frames
     * F     Jacobian matrix of the state transition function with respect to x
     */
    cv::Mat computeProcessNoiseMatrix(double dt, const cv::Mat& F);

    /*
     * Returns the Jacobian matrix of the features measurement function with respect
     * to the complete state (r, q, v, w, f1, ..., fn). This matrix is evaluated at
     * the predicted (a priori) state estimate, therefore it should only be called
     * after applyMotionModel since the latter updates the current estimate x. Also,
     * it updates the vector of predicted insight features pixel positions, inviewPos,
     * and populates a vector with the predicted in sight features indices.
     */
    cv::Mat computeMeasurementMatrix(std::vector<int>& inviewIndices);

    /*
     * Returns the ellipses where the predicted in sight features should be found with
     * high probability. In particular, the ellipses are constructed so that they are
     * confidence regions at level 0.99.
     *
     * means    Predicted in sight features pixel locations
     * S        Innovation covariance matrix of the predicted in sight features
     */
    std::vector<cv::RotatedRect> computeEllipses(const std::vector<cv::Point2d>& means, const cv::Mat& S);

    void drawFeatures(cv::Mat& frame,
                      const std::vector<int>& matchedInviewIndices, const std::vector<int>& failedInviewIndices,
                      const std::vector<cv::RotatedRect>& ellipses, bool drawEllipses = true);

    void removeBadFeatures(const std::vector<int>& failedInviewIndices, const std::vector<int>& failedIndices);

    /*
     * Normalizes the system state orientation quaternion and updates the covariance
     * matrix accordingly by first-order error propagation. It is mandatory that the
     * function be called after the EKF update step since the corrected (a posteriori)
     * orientation quaternion might not be a unit quaternion.
     */
    void renormalizeQuaternion();
};

#endif