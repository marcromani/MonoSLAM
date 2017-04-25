#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>

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

    cout << P << endl;
    cout << H << endl;
    cout << H * P * H.t() << endl << endl;

    exit(0);
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
        F.at<double>(4, 10) = 0.5 * dt;
        F.at<double>(5, 11) = 0.5 * dt;
        F.at<double>(6, 12) = 0.5 * dt;

        // Set D(q_new)/D(O)
        W.at<double>(4, 3) = 0.5 * 0;
        W.at<double>(5, 4) = 0.5 * 0;
        W.at<double>(6, 5) = 0.5 * 0;

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
        F.at<double>(3, 10) = a1 * dq1dw_*w1 - b1 * (dq_dw_*w11 + dt*z) - c1 * dq_dw_*w12 - d1 * dq_dw_*w13;
        F.at<double>(3, 11) = a1 * dq1dw_*w2 - b1 * dq_dw_*w12 - c1 * (dq_dw_*w22 + dt*z) - d1 * dq_dw_*w23;
        F.at<double>(3, 12) = a1 * dq1dw_*w3 - b1 * dq_dw_*w13 - c1 * dq_dw_*w23 - d1 * (dq_dw_*w33 + dt*z);
        F.at<double>(4, 10) = b1 * dq1dw_*w1 + a1 * (dq_dw_*w11 + dt*z) - d1 * dq_dw_*w12 + c1 * dq_dw_*w13;
        F.at<double>(4, 11) = b1 * dq1dw_*w2 + a1 * dq_dw_*w12 - d1 * (dq_dw_*w22 + dt*z) + c1 * dq_dw_*w23;
        F.at<double>(4, 12) = b1 * dq1dw_*w3 + a1 * dq_dw_*w13 - d1 * dq_dw_*w23 + c1 * (dq_dw_*w33 + dt*z);
        F.at<double>(5, 10) = c1 * dq1dw_*w1 + d1 * (dq_dw_*w11 + dt*z) + a1 * dq_dw_*w12 - b1 * dq_dw_*w13;
        F.at<double>(5, 11) = c1 * dq1dw_*w2 + d1 * dq_dw_*w12 + a1 * (dq_dw_*w22 + dt*z) - b1 * dq_dw_*w23;
        F.at<double>(5, 12) = c1 * dq1dw_*w3 + d1 * dq_dw_*w13 + a1 * dq_dw_*w23 - b1 * (dq_dw_*w33 + dt*z);
        F.at<double>(6, 10) = d1 * dq1dw_*w1 - c1 * (dq_dw_*w11 + dt*z) + b1 * dq_dw_*w12 + a1 * dq_dw_*w13;
        F.at<double>(6, 11) = d1 * dq1dw_*w2 - c1 * dq_dw_*w12 + b1 * (dq_dw_*w22 + dt*z) + a1 * dq_dw_*w23;
        F.at<double>(6, 12) = d1 * dq1dw_*w3 - c1 * dq_dw_*w13 + b1 * dq_dw_*w23 + a1 * (dq_dw_*w33 + dt*z);

        // Set D(q_new)/D(O)
        F(Rect(10, 3, 3, 4)).copyTo(W(Rect(3, 3, 3, 4)));
    }

    double r1 = x.at<double>(0, 0);
    double r2 = x.at<double>(1, 0);
    double r3 = x.at<double>(2, 0);
    double q1 = x.at<double>(3, 0);
    double q2 = x.at<double>(4, 0);
    double q3 = x.at<double>(5, 0);
    double q4 = x.at<double>(6, 0);
    double v1 = x.at<double>(7, 0);
    double v2 = x.at<double>(8, 0);
    double v3 = x.at<double>(9, 0);
    double ww1 = x.at<double>(10, 0);
    double ww2 = x.at<double>(11, 0);
    double ww3 = x.at<double>(12, 0);

    for (int r = 1; r <= 13; r++) {
        for (int c = 1; c <= 13; c++) {

            cout << dmodel(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, ww1, ww2, ww3, 0, 0, 0, 0, 0, 0, dt, r, c) << " ";
        }

        cout << endl;
    }

    for (int r = 3; r <= 6; r++) {
        for (int c = 10; c <= 12; c++) {

            F.at<double>(r, c) = dmodel(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, ww1, ww2, ww3, 0, 0, 0, 0, 0, 0, dt, r+1, c+1);
        }
    }

    cout << endl;
    cout << F << endl;
    // Set D(q_new)/D(O)
    F(Rect(10, 3, 3, 4)).copyTo(W(Rect(3, 3, 3, 4)));
}

void Map::computeMeasurementMatrix(Mat& H) {

    // Get the predicted camera rotation (from world basis to camera basis)
    Mat R = getRotationMatrix(x(Rect(0, 3, 1, 4))).t();

    // Get the predicted camera position (in world coordinates)
    Mat t = x(Rect(0, 0, 1, 3));

    vector<Point3d> points3D_(features.size());

    // Set a vector of features positions
    for (unsigned int i = 0; i < points3D_.size(); i++)
        points3D_[i] = Point3d(x(Rect(0, 13 + 3*i, 1, 3)));

    vector<Point2d> points2D_;

    // Project all the features positions to the current view
    camera.projectPoints(R, t, points3D_, points2D_);

    // Features predicted to be in the current view
    vector<reference_wrapper<Feature>> inview;

    // 2D and 3D coordinates of these features
    vector<Point3d> points3D;
    vector<Point2d> points2D;

    Rect box(Point2i(0, 0), camera.frameSize);

    for (unsigned int i = 0; i < points2D_.size(); i++) {

        if (box.contains(points2D_[i])) {

            inview.push_back(ref(features[i]));
            points3D.push_back(points3D_[i]);
            points2D.push_back(points2D_[i]);
        }
    }

    H = Mat(2 * inview.size(), x.rows, CV_64FC1, Scalar(0));

    Mat R1, R2, R3, R4;

    getRotationMatrixDerivatives(x(Rect(0, 3, 1, 4)), R1, R2, R3, R4);

    double fx = camera.K.at<double>(0, 0);
    double fy = camera.K.at<double>(1, 1);

    double k1 = camera.distCoeffs.at<double>(0, 0);
    double k2 = camera.distCoeffs.at<double>(0, 1);
    double p1 = camera.distCoeffs.at<double>(0, 2);
    double p2 = camera.distCoeffs.at<double>(0, 3);
    double k3 = camera.distCoeffs.at<double>(0, 4);

    for (int i = 0; i < H.rows / 2; i++) {

        Mat pt = Mat(points3D[i]) - t;

        Mat pCam = R * pt;

        double xCam = pCam.at<double>(0, 0);
        double yCam = pCam.at<double>(1, 0);
        double zCam = pCam.at<double>(2, 0);

        double zCam2 = zCam * zCam;

        Mat pBar1 = R1 * pt;
        Mat pBar2 = R2 * pt;
        Mat pBar3 = R3 * pt;
        Mat pBar4 = R4 * pt;

        // D(xn)/D(r): derivatives of normalized camera coordinate xn with respect to camera position
        double dxndr1 = (R.at<double>(2, 0) * xCam - R.at<double>(0, 0) * zCam) / zCam2;
        double dxndr2 = (R.at<double>(2, 1) * xCam - R.at<double>(0, 1) * zCam) / zCam2;
        double dxndr3 = (R.at<double>(2, 2) * xCam - R.at<double>(0, 2) * zCam) / zCam2;

        // D(xn)/D(q): derivatives of normalized camera coordinate xn with respect to camera orientation
        double dxndq1 = (pBar1.at<double>(0, 0) * zCam - pBar1.at<double>(2, 0) * xCam) / zCam2;
        double dxndq2 = (pBar2.at<double>(0, 0) * zCam - pBar2.at<double>(2, 0) * xCam) / zCam2;
        double dxndq3 = (pBar3.at<double>(0, 0) * zCam - pBar3.at<double>(2, 0) * xCam) / zCam2;
        double dxndq4 = (pBar4.at<double>(0, 0) * zCam - pBar4.at<double>(2, 0) * xCam) / zCam2;

        // D(yn)/D(r): derivatives of normalized camera coordinate yn with respect to camera position
        double dyndr1 = (R.at<double>(2, 0) * yCam - R.at<double>(1, 0) * zCam) / zCam2;
        double dyndr2 = (R.at<double>(2, 1) * yCam - R.at<double>(1, 1) * zCam) / zCam2;
        double dyndr3 = (R.at<double>(2, 2) * yCam - R.at<double>(1, 2) * zCam) / zCam2;

        // D(yn)/D(q): derivatives of normalized camera coordinate yn with respect to camera orientation
        double dyndq1 = (pBar1.at<double>(1, 0) * zCam - pBar1.at<double>(2, 0) * yCam) / zCam2;
        double dyndq2 = (pBar2.at<double>(1, 0) * zCam - pBar2.at<double>(2, 0) * yCam) / zCam2;
        double dyndq3 = (pBar3.at<double>(1, 0) * zCam - pBar3.at<double>(2, 0) * yCam) / zCam2;
        double dyndq4 = (pBar4.at<double>(1, 0) * zCam - pBar4.at<double>(2, 0) * yCam) / zCam2;

        double xn = xCam / zCam;
        double yn = yCam / zCam;

        // D(r2)/D(r): derivatives of r^2 = xn^2 + yn^2 with respect to camera position
        double dr2dr1 = 2 * (xn * dxndr1 + yn * dyndr1);
        double dr2dr2 = 2 * (xn * dxndr2 + yn * dyndr2);
        double dr2dr3 = 2 * (xn * dxndr3 + yn * dyndr3);

        // D(r2)/D(q): derivatives of r^2 = xn^2 + yn^2 with respect to camera orientation
        double dr2dq1 = 2 * (xn * dxndq1 + yn * dyndq1);
        double dr2dq2 = 2 * (xn * dxndq2 + yn * dyndq2);
        double dr2dq3 = 2 * (xn * dxndq3 + yn * dyndq3);
        double dr2dq4 = 2 * (xn * dxndq4 + yn * dyndq4);

        double r2 = xn * xn + yn * yn;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double a = 1 + k1*r2 + k2*r4 + k3*r6;
        double b = k1 + 2*k2*r2 + 3*k3*r4;

        double ddr1 = 2 * (xn * dyndr1 + yn * dxndr1);
        double ddr2 = 2 * (xn * dyndr2 + yn * dxndr2);
        double ddr3 = 2 * (xn * dyndr3 + yn * dxndr3);

        double ddq1 = 2 * (xn * dyndq1 + yn * dxndq1);
        double ddq2 = 2 * (xn * dyndq2 + yn * dxndq2);
        double ddq3 = 2 * (xn * dyndq3 + yn * dxndq3);
        double ddq4 = 2 * (xn * dyndq4 + yn * dxndq4);

        // D(u)/D(r)
        double dudr1 = fx * (dxndr1*a + xn*dr2dr1*b + p1*ddr1 + p2*(dr2dr1 + 4*xn*dxndr1));
        double dudr2 = fx * (dxndr2*a + xn*dr2dr2*b + p1*ddr2 + p2*(dr2dr2 + 4*xn*dxndr2));
        double dudr3 = fx * (dxndr3*a + xn*dr2dr3*b + p1*ddr3 + p2*(dr2dr3 + 4*xn*dxndr3));

        // D(u)/D(x)
        double dudx1 = - dudr1;
        double dudx2 = - dudr2;
        double dudx3 = - dudr3;

        // D(u)/D(q)
        double dudq1 = fx * (dxndq1*a + xn*dr2dq1*b + p1*ddq1 + p2*(dr2dq1 + 4*xn*dxndq1));
        double dudq2 = fx * (dxndq2*a + xn*dr2dq2*b + p1*ddq2 + p2*(dr2dq2 + 4*xn*dxndq2));
        double dudq3 = fx * (dxndq3*a + xn*dr2dq3*b + p1*ddq3 + p2*(dr2dq3 + 4*xn*dxndq3));
        double dudq4 = fx * (dxndq4*a + xn*dr2dq4*b + p1*ddq4 + p2*(dr2dq4 + 4*xn*dxndq4));

        // D(v)/D(r)
        double dvdr1 = fy * (dyndr1*a + yn*dr2dr1*b + p2*ddr1 + p1*(dr2dr1 + 4*yn*dyndr1));
        double dvdr2 = fy * (dyndr2*a + yn*dr2dr2*b + p2*ddr2 + p1*(dr2dr2 + 4*yn*dyndr2));
        double dvdr3 = fy * (dyndr3*a + yn*dr2dr3*b + p2*ddr3 + p1*(dr2dr3 + 4*yn*dyndr3));

        // D(v)/D(x)
        double dvdx1 = - dvdr1;
        double dvdx2 = - dvdr2;
        double dvdx3 = - dvdr3;

        // D(v)/D(q)
        double dvdq1 = fy * (dyndq1*a + yn*dr2dq1*b + p2*ddq1 + p1*(dr2dq1 + 4*yn*dyndq1));
        double dvdq2 = fy * (dyndq2*a + yn*dr2dq2*b + p2*ddq2 + p1*(dr2dq2 + 4*yn*dyndq2));
        double dvdq3 = fy * (dyndq3*a + yn*dr2dq3*b + p2*ddq3 + p1*(dr2dq3 + 4*yn*dyndq3));
        double dvdq4 = fy * (dyndq4*a + yn*dr2dq4*b + p2*ddq4 + p1*(dr2dq4 + 4*yn*dyndq4));

        H.at<double>(2*i, 0) = dudr1;
        H.at<double>(2*i, 1) = dudr2;
        H.at<double>(2*i, 2) = dudr3;
        H.at<double>(2*i, 3) = dudq1;
        H.at<double>(2*i, 4) = dudq2;
        H.at<double>(2*i, 5) = dudq3;
        H.at<double>(2*i, 6) = dudq4;

        H.at<double>(2*i, 13 + 3*i) = dudx1;
        H.at<double>(2*i, 13 + 3*i+1) = dudx2;
        H.at<double>(2*i, 13 + 3*i+2) = dudx3;

        H.at<double>(2*i+1, 0) = dvdr1;
        H.at<double>(2*i+1, 1) = dvdr2;
        H.at<double>(2*i+1, 2) = dvdr3;
        H.at<double>(2*i+1, 3) = dvdq1;
        H.at<double>(2*i+1, 4) = dvdq2;
        H.at<double>(2*i+1, 5) = dvdq3;
        H.at<double>(2*i+1, 6) = dvdq4;

        H.at<double>(2*i+1, 13 + 3*i) = dvdx1;
        H.at<double>(2*i+1, 13 + 3*i+1) = dvdx2;
        H.at<double>(2*i+1, 13 + 3*i+2) = dvdx3;

        /*cout << dudr1 << " " << du(1, i) << endl;
        cout << dudr2 << " " << du(2, i) << endl;
        cout << dudr3 << " " << du(3, i) << endl;
        cout << dudq1 << " " << du(4, i) << endl;
        cout << dudq2 << " " << du(5, i) << endl;
        cout << dudq3 << " " << du(6, i) << endl;
        cout << dudq4 << " " << du(7, i) << endl;
        cout << dudx1 << " " << du(8, i) << endl;
        cout << dudx2 << " " << du(9, i) << endl;
        cout << dudx3 << " " << du(10, i) << endl;

        cout << dvdr1 << " " << dv(1, i) << endl;
        cout << dvdr2 << " " << dv(2, i) << endl;
        cout << dvdr3 << " " << dv(3, i) << endl;
        cout << dvdq1 << " " << dv(4, i) << endl;
        cout << dvdq2 << " " << dv(5, i) << endl;
        cout << dvdq3 << " " << dv(6, i) << endl;
        cout << dvdq4 << " " << dv(7, i) << endl;
        cout << dvdx1 << " " << dv(8, i) << endl;
        cout << dvdx2 << " " << dv(9, i) << endl;
        cout << dvdx3 << " " << dv(10, i) << endl;*/
    }
}

double Map::u(double r1, double r2, double r3,
              double q1, double q2, double q3, double q4,
              double x1, double x2, double x3) {

    double fx = camera.K.at<double>(0, 0);
    double cx = camera.K.at<double>(0, 2);

    double k1 = camera.distCoeffs.at<double>(0, 0);
    double k2 = camera.distCoeffs.at<double>(0, 1);
    double p1 = camera.distCoeffs.at<double>(0, 2);
    double p2 = camera.distCoeffs.at<double>(0, 3);
    double k3 = camera.distCoeffs.at<double>(0, 4);

    // t
    double r_[] = {r1, r2, r3};
    Mat t(3, 1, CV_64FC1, r_);

    // R
    double q_[] = {q1, q2, q3, q4};
    Mat q(4, 1, CV_64FC1, q_);
    Mat R = getRotationMatrix(q).t();

    // x
    double x_[] = {x1, x2, x3};
    Mat x(3, 1, CV_64FC1, x_);

    Mat xc = R * (x - t);

    double u = xc.at<double>(0, 0) / xc.at<double>(2, 0);
    double v = xc.at<double>(1, 0) / xc.at<double>(2, 0);

    double r22 = u*u + v*v;
    double r4 = r22 * r22;
    double r6 = r4 * r22;

    u = u * (1 + k1*r22 + k2*r4 + k3*r6) + 2*p1*u*v + p2*(r22 + 2*u*u);

    return fx*u + cx;
}

double Map::v(double r1, double r2, double r3,
              double q1, double q2, double q3, double q4,
              double x1, double x2, double x3) {

    double fy = camera.K.at<double>(1, 1);
    double cy = camera.K.at<double>(1, 2);

    double k1 = camera.distCoeffs.at<double>(0, 0);
    double k2 = camera.distCoeffs.at<double>(0, 1);
    double p1 = camera.distCoeffs.at<double>(0, 2);
    double p2 = camera.distCoeffs.at<double>(0, 3);
    double k3 = camera.distCoeffs.at<double>(0, 4);

    // t
    double r_[] = {r1, r2, r3};
    Mat t(3, 1, CV_64FC1, r_);

    // R
    double q_[] = {q1, q2, q3, q4};
    Mat q(4, 1, CV_64FC1, q_);
    Mat R = getRotationMatrix(q).t();

    // x
    double x_[] = {x1, x2, x3};
    Mat x(3, 1, CV_64FC1, x_);

    Mat xc = R * (x - t);

    double u = xc.at<double>(0, 0) / xc.at<double>(2, 0);
    double v = xc.at<double>(1, 0) / xc.at<double>(2, 0);

    double r22 = u*u + v*v;
    double r4 = r22 * r22;
    double r6 = r4 * r22;

    v = v * (1 + k1*r22 + k2*r4 + k3*r6) + 2*p2*u*v + p1*(r22 + 2*v*v);

    return fy*v + cy;
}

double Map::du(int i, int j) {

    double h = 1e-10;

    double r1 = x.at<double>(0, 0);
    double r2 = x.at<double>(1, 0);
    double r3 = x.at<double>(2, 0);
    double q1 = x.at<double>(3, 0);
    double q2 = x.at<double>(4, 0);
    double q3 = x.at<double>(5, 0);
    double q4 = x.at<double>(6, 0);
    double x1 = x.at<double>(13 + 3*j, 0);
    double x2 = x.at<double>(13 + 3*j + 1, 0);
    double x3 = x.at<double>(13 + 3*j + 2, 0);

    if (i == 1)
        return (u(r1 + h, r2, r3, q1, q2, q3, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 2)
        return (u(r1, r2 + h, r3, q1, q2, q3, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 3)
        return (u(r1, r2, r3 + h, q1, q2, q3, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 4)
        return (u(r1, r2, r3, q1 + h, q2, q3, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 5)
        return (u(r1, r2, r3, q1, q2 + h, q3, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 6)
        return (u(r1, r2, r3, q1, q2, q3 + h, q4, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 7)
        return (u(r1, r2, r3, q1, q2, q3, q4 + h, x1, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 8)
        return (u(r1, r2, r3, q1, q2, q3, q4, x1 + h, x2, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 9)
        return (u(r1, r2, r3, q1, q2, q3, q4, x1, x2 + h, x3) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 10)
        return (u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3 + h) - u(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;

    return -1;
}

double Map::dv(int i, int j) {

    double h = 1e-10;

    double r1 = x.at<double>(0, 0);
    double r2 = x.at<double>(1, 0);
    double r3 = x.at<double>(2, 0);
    double q1 = x.at<double>(3, 0);
    double q2 = x.at<double>(4, 0);
    double q3 = x.at<double>(5, 0);
    double q4 = x.at<double>(6, 0);
    double x1 = x.at<double>(13 + 3*j, 0);
    double x2 = x.at<double>(13 + 3*j + 1, 0);
    double x3 = x.at<double>(13 + 3*j + 2, 0);

    if (i == 1)
        return (v(r1 + h, r2, r3, q1, q2, q3, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 2)
        return (v(r1, r2 + h, r3, q1, q2, q3, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 3)
        return (v(r1, r2, r3 + h, q1, q2, q3, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 4)
        return (v(r1, r2, r3, q1 + h, q2, q3, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 5)
        return (v(r1, r2, r3, q1, q2 + h, q3, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 6)
        return (v(r1, r2, r3, q1, q2, q3 + h, q4, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 7)
        return (v(r1, r2, r3, q1, q2, q3, q4 + h, x1, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 8)
        return (v(r1, r2, r3, q1, q2, q3, q4, x1 + h, x2, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 9)
        return (v(r1, r2, r3, q1, q2, q3, q4, x1, x2 + h, x3) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;
    if (i == 10)
        return (v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3 + h) - v(r1, r2, r3, q1, q2, q3, q4, x1, x2, x3)) / h;

    return -1;
}

double Map::model(double r1, double r2, double r3,
                  double q1, double q2, double q3, double q4,
                  double v1, double v2, double v3,
                  double w1, double w2, double w3,
                  double V1, double V2, double V3,
                  double W1, double W2, double W3,
                  double dt,
                  int i) {

    // r
    double r_[] = {r1, r2, r3};
    Mat r(3, 1, CV_64FC1, r_);

    // q
    double q_[] = {q1, q2, q3, q4};
    Mat q(4, 1, CV_64FC1, q_);

    // v
    double v_[] = {v1, v2, v3};
    Mat v(3, 1, CV_64FC1, v_);

    // w
    double w_[] = {w1, w2, w3};
    Mat w(3, 1, CV_64FC1, w_);

    // V
    double V_[] = {V1, V2, V3};
    Mat V(3, 1, CV_64FC1, V_);

    // W
    double W_[] = {W1, W2, W3};
    Mat W(3, 1, CV_64FC1, W_);

    r += (v + V) * dt;

    if (w1 != 0 || w2 != 0 || w3 != 0)
        q = quaternionMultiply(
                q,
                computeQuaternion((w + W) * dt)
            );

    v += V;
    w += W;

    if (i == 1)
        return r.at<double>(0, 0);
    if (i == 2)
        return r.at<double>(1, 0);
    if (i == 3)
        return r.at<double>(2, 0);
    if (i == 4)
        return q.at<double>(0, 0);
    if (i == 5)
        return q.at<double>(1, 0);
    if (i == 6)
        return q.at<double>(2, 0);
    if (i == 7)
        return q.at<double>(3, 0);
    if (i == 8)
        return v.at<double>(0, 0);
    if (i == 9)
        return v.at<double>(1, 0);
    if (i == 10)
        return v.at<double>(2, 0);
    if (i == 11)
        return w.at<double>(0, 0);
    if (i == 12)
        return w.at<double>(1, 0);
    if (i == 13)
        return w.at<double>(2, 0);
    if (i == 14)
        return V.at<double>(0, 0);
    if (i == 15)
        return V.at<double>(1, 0);
    if (i == 16)
        return V.at<double>(2, 0);
    if (i == 17)
        return W.at<double>(0, 0);
    if (i == 18)
        return W.at<double>(1, 0);
    if (i == 19)
        return W.at<double>(2, 0);

    return -1;
}

double Map::dmodel(double r1, double r2, double r3,
                   double q1, double q2, double q3, double q4,
                   double v1, double v2, double v3,
                   double w1, double w2, double w3,
                   double V1, double V2, double V3,
                   double W1, double W2, double W3,
                   double dt,
                   int i, int j) {

    double h = 1e-3;

    if (j == 1)
        return (model(r1 + h, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 2)
        return (model(r1, r2 + h, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 3)
        return (model(r1, r2, r3 + h, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 4)
        return (model(r1, r2, r3, q1 + h, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 5)
        return (model(r1, r2, r3, q1, q2 + h, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 6)
        return (model(r1, r2, r3, q1, q2, q3 + h, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 7)
        return (model(r1, r2, r3, q1, q2, q3, q4 + h, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 8)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1 + h, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 9)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2 + h, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 10)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3 + h, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 11)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1 + h, w2, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 12)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2 + h, w3, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 13)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3 + h, V1, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 14)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1 + h, V2, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 15)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2 + h, V3, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 16)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3 + h, W1, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 17)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1 + h, W2, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 18)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2 + h, W3, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    if (j == 19)
        return (model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3 + h, dt, i) -
                model(r1, r2, r3, q1, q2, q3, q4, v1, v2, v3, w1, w2, w3, V1, V2, V3, W1, W2, W3, dt, i)) / h;

    return -1;
}