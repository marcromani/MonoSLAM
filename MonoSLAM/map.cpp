#include <algorithm>
#include <functional>
#include <iostream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "map.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

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
bool Map::initMap(const Mat& frame, const Size& patternSize, double squareSize,
                  const vector<double>& var) {

    vector<Point2f> imageCorners;

    bool patternFound = findChessboardCorners(frame, patternSize, imageCorners,
                        CALIB_CB_ADAPTIVE_THRESH +
                        CALIB_CB_NORMALIZE_IMAGE +
                        CALIB_CB_FAST_CHECK);

    if (!patternFound)
        return false;

    cornerSubPix(frame, imageCorners, Size(11, 11), Size(-1, -1),
                 TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    vector<Point3f> worldCorners;

    // Set the world reference frame (y axis along pattern width)
    for (int i = 0; i < patternSize.height; i++)
        for (int j = 0; j < patternSize.width; j++)
            worldCorners.push_back(Point3d(squareSize * i, squareSize * j, 0));

    Mat rvec, tvec;

    // Find camera pose
    bool poseFound = solvePnP(worldCorners, imageCorners, camera.K, camera.distCoeffs, rvec, tvec,
                              false, SOLVEPNP_ITERATIVE);

    if (!poseFound)
        return false;

    Mat R, t;

    // Convert to rotation R (from world to camera) and translation t (in world coordinates)
    Rodrigues(rvec, R);
    t = - R.t() * tvec;

    // Compute the camera orientation quaternion
    Mat q = quaternionInv(computeQuaternion(rvec));

    // Update the map state vector with the camera state vector (zero linear and angular velocities)
    t.copyTo(x(Rect(0, 0, 1, 3)));
    q.copyTo(x(Rect(0, 3, 1, 4)));

    // Update the map covariance matrix with the camera covariance matrix
    P = Mat::diag(Mat(var));

    // Extract the chessboard outer corners and add the initial features
    int idx;
    bool init = true;

    idx = 0;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = patternSize.width - 1;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = (patternSize.height - 1) * patternSize.width;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = patternSize.width * patternSize.height - 1;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    if (!init) {

        reset();
        return false;
    }

    return true;
}

/*
 * Updates the camera and features data with the map state vector and map state
 * covariance matrix data. Only shallow copies are performed. This function should
 * be called whenever the current state of the system must be broadcast to each
 * of its parts. This method exists because the map state (x and P) is continuously
 * updated by the EKF and these changes will not be directly reflected in the camera
 * and features instances.
 */
void Map::broadcastData() {

    camera.r = x(Rect(0, 0, 1, 3));
    camera.q = x(Rect(0, 3, 1, 4));
    camera.v = x(Rect(0, 7, 1, 3));
    camera.w = x(Rect(0, 10, 1, 3));

    camera.R = getRotationMatrix(camera.q);

    camera.P = P(Rect(0, 0, 13, 13));

    for (unsigned int i = 0; i < features.size(); i++) {

        features[i].pos = x(Rect(0, 13 + 3*i, 1, 3));
        features[i].P = P(Rect(13 + 3*i, 13 + 3*i, 3, 3));
    }
}

/*
 * Detects corners and initializes new feature candidates whenever there are not
 * enough visible features and no candidates are being tracked. Returns a boolean
 * indicating whether new candidates were detected or not.
 *
 * frame    Grayscale frame
 */
bool Map::trackNewCandidates(const Mat& frame) {

    // If there are enough visible features there is no need to detect new ones
    if (numVisibleFeatures >= minFeatureDensity)
        return false;

    // The same applies if there are few features but candidates are already being tracked
    if (!candidates.empty())
        return false;

    Mat corners = findCorners(frame);

    // If no good corners were found just pass
    if (corners.empty())
        return false;

    // Undistort and normalize the pixels
    Mat undistorted;
    undistortPoints(corners, undistorted, camera.K, camera.distCoeffs);

    // Add the z component and reshape to 3xN
    undistorted = undistorted.reshape(1).t();
    undistorted.resize(3, Scalar(1));

    // Get camera translation and rotation
    Mat t = x(Rect(0, 0, 1, 3));
    Mat R = getRotationMatrix(x(Rect(0, 3, 1, 4)));

    // Compute the directions (in world coordinates) of the lines that contain the features
    undistorted.convertTo(undistorted, CV_64FC1);
    Mat directions = R * undistorted;

    Point2d depthInterval(0.05, 5);
    int depthSamples = 100;

    // Normalize the direction vectors and add new pre-initialized features
    for (int i = 0; i < directions.cols; i++) {

        normalize(directions.col(i), directions.col(i));

        Point2i pixel(corners.at<Point2f>(i));
        candidates.push_back(Feature(frame, buildSquare(pixel, patchSize), R.t(), t,
                                     directions.col(i), depthInterval, depthSamples));
    }

    return true;
}

/*
 * Draws features that are (theoretically) in view.
 *
 * frame    Camera frame
 */
void Map::drawInViewFeatures(const Mat& frame) {

    for (unsigned int i = 0; i < inview.size(); i++)
        circle(frame, inview[i], 5, Scalar(0, 0, 255), 1, CV_AA, 0);
}

/*
 * Applies the first step of the Extended Kalman Filter. In particular, given the
 * current state estimate x_k|k and the current state covariance matrix P_k|k, it
 * computes the next a priori state estimate x_k+1|k (via the camera motion model)
 * and its associated covariance matrix P_k+1|k.
 *
 * dt    Time interval between the current and past frames
 */
void Map::predict(double dt) {

    Mat F, W;

    computePredictionMatrices(dt, F, W);
    applyMotionModel(dt);

    P = F * P * F.t() + W * A*dt*dt * W.t();

    Mat H;
    computeMeasurementMatrix(H);
}

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
bool Map::addInitialFeature(const Mat& frame, const Point2f& pos2D, const Point3f& pos3D,
                            const Mat& R, const Mat& t) {

    // Resize the state vector to accommodate the new feature position
    x.resize(x.rows + 3);

    // Copy the new feature position into the state vector
    double pos3D_[] = {pos3D.x, pos3D.y, pos3D.z};
    Mat(3, 1, CV_64FC1, pos3D_).copyTo(x(Rect(0, x.rows - 3, 1, 3)));

    // Resize the state covariance matrix
    Mat tmp(P.rows + 3, P.cols + 3, CV_64FC1, Scalar(0));
    P.copyTo(tmp(Rect(0, 0, P.cols, P.rows)));
    P = tmp;

    // Compute the feature patch normal
    double n[] = {0, 0, 1};
    Mat normal(3, 1, CV_64FC1, n);

    Point2i pixelPosition(pos2D);
    Rect roi = buildSquare(pixelPosition, patchSize);

    bool overflow = roi.x < 0 || roi.x + roi.width > frame.cols ||
                    roi.y < 0 || roi.y + roi.height > frame.rows;

    // Check if the feature patch is outside the frame boundaries
    if (overflow)
        return false;

    // Add the feature
    features.push_back(Feature(frame, roi, normal, R, t));

    // Add its image position to the list of in view features
    inview.push_back(pos2D);

    numVisibleFeatures++;

    return true;
}

/*
 * Sets the map as it was upon creation.
 */
void Map::reset() {

    features.clear();
    candidates.clear();
    inview.clear();

    numVisibleFeatures = 0;

    x = Mat(13, 1, CV_64FC1, Scalar(0));
    P = Mat(13, 13, CV_64FC1, Scalar(0));
}

/*
 * Returns corners found by the Shi-Tomasi corner detector. In particular, the
 * number of computed corners is such that when the associated pre-initialized
 * features are added to the map for tracking, the number of visible features
 * in the current frame will (hopefully) be the maximum allowed.
 *
 * frame    Grayscale frame
 */
Mat Map::findCorners(const Mat& frame) {

    Mat corners;

    int maxCorners = maxFeatureDensity - numVisibleFeatures;
    int minDistance = 60;
    double qualityLevel = 0.2;

    Mat mask(frame.size(), CV_8UC1, Scalar(0));

    // Set a margin around the mask to enclose the detected corners inside
    // a rectangle and avoid premature corner loss due to camera movement.
    // Pad should be larger than patchSize/2 so that new features patches
    // fit inside the camera frame.
    int pad = 30;
    mask(Rect(pad, pad, mask.cols - 2 * pad, mask.rows - 2 * pad)).setTo(Scalar(255));

    for (unsigned int i = 0; i < inview.size(); i++) {

        Rect roi = buildSquare(inview[i], 2 * minDistance);

        roi.x = max(0, roi.x);
        roi.y = max(0, roi.y);

        if (roi.x + roi.width > frame.cols)
            roi.width = frame.cols - roi.x;
        if (roi.y + roi.height > frame.rows)
            roi.height = frame.rows - roi.y;

        // Update the mask to reject the region around the ith inview feature
        mask(roi).setTo(Scalar(0));
    }

    goodFeaturesToTrack(frame, corners, maxCorners, qualityLevel, minDistance, mask);

    return corners;
}

/*
 * Applies the state transition function to the current state estimate. It is
 * mandatory that this method be called after computePredictionMatrices, since
 * the latter expects the last a posteriori (corrected) state estimate in the
 * state vector x.
 *
 * dt    Time interval between the current and past frames
 */
void Map::applyMotionModel(double dt) {

    x(Rect(0, 0, 1, 3)) += x(Rect(0, 7, 1, 3)) * dt;

    if (x.at<double>(10, 0) != 0 || x.at<double>(11, 0) != 0 || x.at<double>(12, 0) != 0)
        x(Rect(0, 3, 1, 4)) = quaternionMultiply(
                                  x(Rect(0, 3, 1, 4)),
                                  computeQuaternion(x(Rect(0, 10, 1, 3)) * dt)
                              );
}

/*
 * Computes the Jacobian matrices of the state transition function with respect to
 * the state (r, q, v, w, f1, ..., fn) and to the noise (V1, V2, V3, O1, O2, O3).
 * These matrices are evaluated at the current state estimate on the assumption that
 * there is zero process noise. This method must be called before applyMotionModel
 * since the latter updates the current state estimate x.
 *
 * dt    Time interval between the current and past frames
 * F     Jacobian matrix of the state transition function with respect to x
 * W     Jacobian matrix of the state transition function with respect to noise
 */
void Map::computePredictionMatrices(double dt, Mat& F, Mat& W) {

    F = Mat::eye(P.size(), CV_64FC1);
    W = Mat(P.rows, 6, CV_64FC1, Scalar(0));

    // Set D(r_new)/D(v_old)
    F.at<double>(0, 7) = dt;
    F.at<double>(1, 8) = dt;
    F.at<double>(2, 9) = dt;

    // Set D(r_new)/D(V)
    W.at<double>(0, 0) = dt;
    W.at<double>(1, 1) = dt;
    W.at<double>(2, 2) = dt;

    // Set D(v_new)/D(V) and D(w_new)/D(O)
    W.at<double>(7, 0) = 1;
    W.at<double>(8, 1) = 1;
    W.at<double>(9, 2) = 1;
    W.at<double>(10, 3) = 1;
    W.at<double>(11, 4) = 1;
    W.at<double>(12, 5) = 1;

    double w1 = x.at<double>(10, 0);
    double w2 = x.at<double>(11, 0);
    double w3 = x.at<double>(12, 0);

    if (w1 == 0 && w2 == 0 && w3 == 0) {

        // Set D(q_new)/D(w_old)
        F.at<double>(4, 10) = 0.5;
        F.at<double>(5, 11) = 0.5;
        F.at<double>(6, 12) = 0.5;

        // Set D(q_new)/D(O)
        W.at<double>(4, 3) = 0.5;
        W.at<double>(5, 4) = 0.5;
        W.at<double>(6, 5) = 0.5;

    } else {

        Mat q = computeQuaternion(x(Rect(0, 10, 1, 3)) * dt);

        double a2 = q.at<double>(0, 0);
        double b2 = q.at<double>(1, 0);
        double c2 = q.at<double>(2, 0);
        double d2 = q.at<double>(3, 0);

        // Set D(q_new)/D(q_old)
        F.at<double>(3, 3) =  a2;
        F.at<double>(3, 4) = -b2;
        F.at<double>(3, 5) = -c2;
        F.at<double>(3, 6) = -d2;
        F.at<double>(4, 3) =  b2;
        F.at<double>(4, 4) =  a2;
        F.at<double>(4, 5) =  d2;
        F.at<double>(4, 6) = -c2;
        F.at<double>(5, 3) =  c2;
        F.at<double>(5, 4) = -d2;
        F.at<double>(5, 5) =  a2;
        F.at<double>(5, 6) =  b2;
        F.at<double>(6, 3) =  d2;
        F.at<double>(6, 4) =  c2;
        F.at<double>(6, 5) = -b2;
        F.at<double>(6, 6) =  a2;

        double w11 = w1 * w1;
        double w22 = w2 * w2;
        double w33 = w3 * w3;

        double norm2 = w11 + w22 + w33;
        double norm = sqrt(norm2);

        double c = cos(0.5 * dt * norm);
        double s = sin(0.5 * dt * norm);

        double dq1dw_ = - 0.5 * dt * s / norm;
        double z = s / (dt * norm);
        double dq_dw_ = 0.5 * c / norm2 - z / norm2;

        double w12 = w1 * w2;
        double w13 = w1 * w3;
        double w23 = w2 * w3;

        double a1 = x.at<double>(3, 0);
        double b1 = x.at<double>(4, 0);
        double c1 = x.at<double>(5, 0);
        double d1 = x.at<double>(6, 0);

        // Set D(q_new)/D(w_old)
        F.at<double>(3, 10) = a1 * dq1dw_*w1 - b1 * (dq_dw_*w11 + z) - c1 * dq_dw_*w12 - d1 * dq_dw_*w13;
        F.at<double>(3, 11) = a1 * dq1dw_*w2 - b1 * dq_dw_*w12 - c1 * (dq_dw_*w22 + z) - d1 * dq_dw_*w23;
        F.at<double>(3, 12) = a1 * dq1dw_*w3 - b1 * dq_dw_*w13 - c1 * dq_dw_*w23 - d1 * (dq_dw_*w33 + z);
        F.at<double>(4, 10) = b1 * dq1dw_*w1 + a1 * (dq_dw_*w11 + z) - d1 * dq_dw_*w12 + c1 * dq_dw_*w13;
        F.at<double>(4, 11) = b1 * dq1dw_*w2 + a1 * dq_dw_*w12 - d1 * (dq_dw_*w22 + z) + c1 * dq_dw_*w23;
        F.at<double>(4, 12) = b1 * dq1dw_*w3 + a1 * dq_dw_*w13 - d1 * dq_dw_*w23 + c1 * (dq_dw_*w33 + z);
        F.at<double>(5, 10) = c1 * dq1dw_*w1 + d1 * (dq_dw_*w11 + z) + a1 * dq_dw_*w12 - b1 * dq_dw_*w13;
        F.at<double>(5, 11) = c1 * dq1dw_*w2 + d1 * dq_dw_*w12 + a1 * (dq_dw_*w22 + z) - b1 * dq_dw_*w23;
        F.at<double>(5, 12) = c1 * dq1dw_*w3 + d1 * dq_dw_*w13 + a1 * dq_dw_*w23 - b1 * (dq_dw_*w33 + z);
        F.at<double>(6, 10) = d1 * dq1dw_*w1 - c1 * (dq_dw_*w11 + z) + b1 * dq_dw_*w12 + a1 * dq_dw_*w13;
        F.at<double>(6, 11) = d1 * dq1dw_*w2 - c1 * dq_dw_*w12 + b1 * (dq_dw_*w22 + z) + a1 * dq_dw_*w23;
        F.at<double>(6, 12) = d1 * dq1dw_*w3 - c1 * dq_dw_*w13 + b1 * dq_dw_*w23 + a1 * (dq_dw_*w33 + z);

        // Set D(q_new)/D(O)
        F(Rect(10, 3, 3, 4)).copyTo(W(Rect(3, 3, 3, 4)));
    }
}

void Map::computeMeasurementMatrix(Mat& H) {

    // Get the predicted camera rotation (from world basis to camera basis)
    Mat R = getRotationMatrix(x(Rect(0, 3, 1, 4))).t();

    // Get the predicted camera position (in world coordinates)
    Mat t = x(Rect(0, 0, 1, 3));

    vector<Point3d> points3D(features.size());

    // Set a vector of features positions
    for (unsigned int i = 0; i < points3D.size(); i++)
        points3D[i] = Point3d(x(Rect(0, 13 + 3*i, 1, 3)));

    vector<Point2d> points2D;

    // Project all the features positions to the current view
    camera.projectPoints(R, t, points3D, points2D);

    // Features predicted to be in the current view
    vector<reference_wrapper<Feature>> inview;

    Rect box(Point2i(0, 0), camera.frameSize);

    for (unsigned int i = 0; i < points2D.size(); i++)
        if (box.contains(points2D[i]))
            inview.push_back(ref(features[i]));

    H = Mat(2 * inview.size(), x.rows, CV_64FC1, Scalar(0));
}