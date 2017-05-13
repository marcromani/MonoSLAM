#include <algorithm>
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

    int w = patternSize.width;
    int h = patternSize.height;

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
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
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

    // Extract the chessboard outer and central corners and add the initial features
    int idx;
    bool init = true;

    idx = 0;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = w - 1;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = (h - 1) * w;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = w * h - 1;
    init = init && addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = w * ((h - 1) / 2) + (w + 1) / 2 - 1;
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

    // Undistort pixels and compute their depth-normalized (z = 1) camera coordinates
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

    Point2d depthInterval(0.2, 3);
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
 * Applies the first step of the Extended Kalman Filter. In particular, given the
 * current state estimate x_k|k and the current state covariance matrix P_k|k, it
 * computes the next a priori state estimate x_k+1|k (via the camera motion model)
 * and its associated covariance matrix P_k+1|k.
 *
 * dt    Time interval between the current and past frames
 */
void Map::predict(double dt) {

    Mat F = computeProcessMatrix(dt);
    Mat N = computeProcessNoiseMatrix(dt, F);

    P = F * P * F.t() + N;

    applyMotionModel(dt);
}

/*
 * Applies the update step of the Extended Kalman Filter. In particular, given the
 * predicted (a priori) state estimate x_k+1|k and the associated covariance matrix
 * P_k+1|k, it computes the corrected (a posteriori) estimate x_k+1|k+1 and its
 * covariance P_k+1|k+1 by measuring known, currently visible map features.
 *
 * gray     Grayscale frame, used for feature matching
 * frame    Color frame, used as a canvas to draw features on
 */
void Map::update(const Mat& gray, Mat& frame) {

    vector<int> inviewIndices;

    Mat H = computeMeasurementMatrix(inviewIndices);

    // If there are no predicted in sight features skip the update step
    if (inviewIndices.empty()) {

        numVisibleFeatures = 0;
        return;
    }

    Mat N(H.rows, H.rows, CV_64FC1, Scalar(0));

    for (int i = 0; i < N.rows / 2; i++) {

        N.at<double>(2*i, 2*i) = R.at<double>(0, 0);
        N.at<double>(2*i + 1, 2*i + 1) = R.at<double>(1, 0);
    }

    // Compute the innovation covariance matrix
    Mat S = H * P * H.t() + N;

    // Compute the error ellipses of the predicted in sight features pixel positions
    vector<RotatedRect> ellipses = computeEllipses(inviewPos, S);

    vector<int> matchedInviewIndices;   // Positions of inviewIndices that contain matched features indices
    vector<int> failedInviewIndices;    // Positions of inviewIndices that contain failed features indices

    vector<int> failedIndices;          // Subset of inviewIndices: not matched features indices

    vector<Point2d> residuals;

    Feature *featuresPtr = features.data();

    // For each predicted in sight feature
    for (unsigned int i = 0; i < ellipses.size(); i++) {

        int idx = inviewIndices[i];

        Feature *feature = &featuresPtr[idx];

        feature->matchingAttempts++;

        Mat p = x(Rect(0, 13 + 3*idx, 1, 3));
        Mat n = feature->normal;
        Mat view1 = feature->image;
        Rect patch1 = feature->roi;
        Mat R1 = feature->R;
        Mat t1 = feature->t;
        Mat R2 = getRotationMatrix(x(Rect(0, 3, 1, 4))).t();
        Mat t2 = x(Rect(0, 0, 1, 3));

        // Compute its appearance from the current camera pose
        int u, v;
        Mat mask;
        Mat templ = camera.warpPatch(p, n, view1, patch1, R1, t1, R2, t2, u, v, mask);

        // Show the warped template at the predicted in sight feature position
        Mat colorTempl;
        cvtColor(templ, colorTempl, CV_GRAY2RGB);
        drawTemplate(frame, colorTempl, inviewPos[i], u, v);

        // If the template size is zero the feature is far away and not visible
        if (templ.empty()) {

            feature->matchingFails++;
            failedInviewIndices.push_back(i);
            failedIndices.push_back(idx);

        } else {

            // Compute the rectangular region to match this template
            Rect roi = getBoundingBox(ellipses[i], gray.size());

            // Get the corresponding image
            Mat image = computeMatchingImage(gray, templ, u, v, roi);

            // Match the template
            Mat ccorr;
            matchTemplate(image, templ, ccorr, TM_CCORR_NORMED, mask);

            // Get the observation value
            double maxVal;
            Point2i maxLoc;
            minMaxLoc(ccorr, 0, &maxVal, 0, &maxLoc);

            if (maxVal > 0.94) {

                int px = maxLoc.x + roi.x + u;
                int py = maxLoc.y + roi.y + v;

                residuals.push_back(Point2d(px - inviewPos[i].x, py - inviewPos[i].y));

                matchedInviewIndices.push_back(i);

            } else {

                feature->matchingFails++;
                failedInviewIndices.push_back(i);
                failedIndices.push_back(idx);
            }
        }
    }

    // Update the number of visible features
    numVisibleFeatures = matchedInviewIndices.size();

    if (numVisibleFeatures == 0) {

        drawFeatures(frame, matchedInviewIndices, failedInviewIndices, ellipses);
        removeBadFeatures(failedInviewIndices, failedIndices);
        return;
    }

    // Compute measurement residual
    Mat y = Mat(residuals).t();
    y = y.reshape(1).t();

    if (!failedInviewIndices.empty()) {

        // Reshape measurement matrix H and innovation covariance S
        H = removeRows(H, failedInviewIndices, 0, 2);
        S = removeRowsCols(S, failedInviewIndices, 0, 2);
    }

    // Compute Kalman gain
    Mat K = P * H.t() * S.inv();

    // Update the map state and the map state covariance matrix
    x += K * y;
    P -= K * H * P;

    drawFeatures(frame, matchedInviewIndices, failedInviewIndices, ellipses);
    removeBadFeatures(failedInviewIndices, failedIndices);
    renormalizeQuaternion();
}

void Map::updateCandidates(const Mat& gray, Mat& frame, atomic<bool>& threadCompleted) {

    threadCompleted.store(false);

    // Get the current (a posteriori) camera rotation (world to camera) and position (in world coordinates)
    Mat R = getRotationMatrix(x(Rect(0, 3, 1, 4))).t();
    Mat t = x(Rect(0, 0, 1, 3));

    Mat R1, R2, R3, R4;
    getRotationMatrixDerivatives(x(Rect(0, 3, 1, 4)), R1, R2, R3, R4);

    double fx = camera.K.at<double>(0, 0);
    double fy = camera.K.at<double>(1, 1);

    double k1 = camera.distCoeffs.at<double>(0, 0);
    double k2 = camera.distCoeffs.at<double>(0, 1);
    double p1 = camera.distCoeffs.at<double>(0, 2);
    double p2 = camera.distCoeffs.at<double>(0, 3);
    double k3 = camera.distCoeffs.at<double>(0, 4);

    Feature *candidatesPtr = candidates.data();

    // For each feature candidate
    for (unsigned int i = 0; i < candidates.size(); i++) {

        Feature *candidate = &candidatesPtr[i];

        vector<Point3d> points3D_(candidate->depths.size());

        // Set a vector of positions hypotheses
        for (unsigned int j = 0; j < points3D_.size(); j++) {

            double d = candidate->depths[j];

            Mat x = candidate->t + d * candidate->dir;
            points3D_[j] = Point3d(x);
        }

        vector<Point2d> points2D_;

        // Project all the positions hypotheses to the current view
        camera.projectPoints(R, t, points3D_, points2D_);

        vector<Point2d> inviewHypothesesPos2D(points2D_.size());
        vector<Point3d> inviewHypothesesPos3D(points2D_.size());
        vector<int> failedIndices(points2D_.size());

        Rect box(Point2i(0, 0), camera.frameSize);

        unsigned int k = 0;

        for (unsigned int j = 0; j < points2D_.size(); j++) {

            if (box.contains(points2D_[j])) {

                inviewHypothesesPos2D[k] = points2D_[j];
                inviewHypothesesPos3D[k] = points3D_[j];

                k++;

            } else {

                failedIndices[j - k] = j;
            }
        }

        inviewHypothesesPos2D.resize(k);
        inviewHypothesesPos3D.resize(k);
        failedIndices.resize(points2D_.size() - k);

        removeIndices(candidate->depths, failedIndices);
        candidate->probs = removeRows(candidate->probs, failedIndices);

        Mat S(2*k, 2*k, CV_64FC1);

        // Derivative of the observation u, v with respect to the camera state (without v and w)
        Mat DuvDx(2, 7, CV_64FC1);

        // Derivative of the observation u, v with respect to the hypothesis
        Mat DuvDy(2, 3, CV_64FC1);

        Mat Pxx = P(Rect(0, 0, 7, 7));
        Mat Pyy = P(Rect(0, 0, 3, 3));

        // Compute the covariance matrices of the projections of the hypotheses in view
        for (unsigned int j = 0; j < k; j++) {

            Mat pt = Mat(inviewHypothesesPos3D[j]) - t;

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

            DuvDx.at<double>(0, 0) = dudr1;
            DuvDx.at<double>(0, 1) = dudr2;
            DuvDx.at<double>(0, 2) = dudr3;
            DuvDx.at<double>(0, 3) = dudq1;
            DuvDx.at<double>(0, 4) = dudq2;
            DuvDx.at<double>(0, 5) = dudq3;
            DuvDx.at<double>(0, 6) = dudq4;

            DuvDx.at<double>(1, 0) = dvdr1;
            DuvDx.at<double>(1, 1) = dvdr2;
            DuvDx.at<double>(1, 2) = dvdr3;
            DuvDx.at<double>(1, 3) = dvdq1;
            DuvDx.at<double>(1, 4) = dvdq2;
            DuvDx.at<double>(1, 5) = dvdq3;
            DuvDx.at<double>(1, 6) = dvdq4;

            DuvDy.at<double>(0, 0) = dudx1;
            DuvDy.at<double>(0, 1) = dudx2;
            DuvDy.at<double>(0, 2) = dudx3;

            DuvDy.at<double>(1, 0) = dvdx1;
            DuvDy.at<double>(1, 1) = dvdx2;
            DuvDy.at<double>(1, 2) = dvdx3;

            S(Rect(2*j, 2*j, 2, 2)) = DuvDx * Pxx * DuvDx.t() +
                                      DuvDy * Pyy * DuvDy.t() + 9 * Mat::eye(2, 2, CV_64FC1);
        }

        vector<RotatedRect> ellipses = computeEllipses(inviewHypothesesPos2D, S);

        // Black background
        Mat black(gray.size(), gray.type(), Scalar::all(0));

        // Copy the search region of each ellipse to the black background
        for (unsigned int j = 0; j < k; j++) {

            Rect roi = getBoundingBox(ellipses[j], gray.size());
            gray(roi).copyTo(black(roi));
        }

        // Get the candidate template
        Mat templ = candidate->image(candidate->roi);

        // Get the image to match the template to
        int u = candidate->roi.width / 2;
        int v = candidate->roi.height / 2;
        Rect roi(Point2i(0, 0), black.size());
        Mat image = computeMatchingImage(black, templ, u, v, roi);

        // Match the template
        Mat ccorr;
        matchTemplate(image, templ, ccorr, TM_CCORR_NORMED);

        // Get the observation value
        double maxVal;
        Point2i maxLoc;
        minMaxLoc(ccorr, 0, &maxVal, 0, &maxLoc);

        int px = maxLoc.x + roi.x + u;
        int py = maxLoc.y + roi.y + v;

        Mat x = (Mat_<double>(2, 1) << px, py);

        for (unsigned int j = 0; j < k; j++) {

            Mat mean = Mat(inviewHypothesesPos2D[j]);
            candidate->probs.at<double>(j, 0) *= gaussian2Dpdf(x, mean, S(Rect(2*j, 2*j, 2, 2)));
        }

        candidate->probs /= sum(candidate->probs)[0];
    }

    for (int i = candidates.size() - 1; i >= 0; i--) {

        Mat depths = Mat(candidates[i].depths);
        Mat probs = candidates[i].probs;

        cout << depths.size() << " " << probs.size() << endl;

        double mean = depths.dot(probs);

        Mat diff = depths - Mat(depths.rows, 1, CV_64FC1, Scalar::all(mean));
        Mat diff2 = diff.mul(diff);

        double stdDev = sqrt(diff2.dot(probs));

        if (stdDev / mean < 0.3) {

            /*for (int j = 0; j < depths.rows; j++)
                cout << depths.at<double>(j, 0) << " " << probs.at<double>(j, 0) << endl;
            cout << endl;*/

            candidates.erase(candidates.begin() + i);
        }
    }

    threadCompleted.store(true);
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
    inviewPos.push_back(pos2D);

    numVisibleFeatures++;

    return true;
}

/*
 * Sets the map as it was upon creation.
 */
void Map::reset() {

    features.clear();
    candidates.clear();
    inviewPos.clear();

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

    for (unsigned int i = 0; i < inviewPos.size(); i++) {

        Rect roi = buildSquare(inviewPos[i], 2 * minDistance);

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
 * Applies the state transition function to the current state estimate. It is mandatory
 * that this method be called after computeProcessMatrix and computeProcessNoiseMatrix
 * since the latter expect the last a posteriori (corrected) state estimate in the state
 * vector x.
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
 * Returns the Jacobian matrix of the state transition function with respect to the
 * complete state (r, q, v, w, f1, ..., fn). This matrix is evaluated at the current
 * state estimate on the assumption that there is zero process noise. This method must
 * be called before applyMotionModel since the latter updates the current estimate x.
 *
 * dt    Time interval between the current and past frames
 */
Mat Map::computeProcessMatrix(double dt) {

    Mat F = Mat::eye(P.size(), CV_64FC1);

    // Set D(r_new)/D(v_old)

    F.at<double>(0, 7) = dt;
    F.at<double>(1, 8) = dt;
    F.at<double>(2, 9) = dt;

    // Compute D(q_new/q(w*dt))

    Mat Q1(4, 4, CV_64FC1);

    // Current pose quaternion (q_old)
    double a1 = x.at<double>(3, 0);
    double b1 = x.at<double>(4, 0);
    double c1 = x.at<double>(5, 0);
    double d1 = x.at<double>(6, 0);

    // Row 1
    Q1.at<double>(0, 0) =  a1;
    Q1.at<double>(0, 1) = -b1;
    Q1.at<double>(0, 2) = -c1;
    Q1.at<double>(0, 3) = -d1;

    // Row 2
    Q1.at<double>(1, 0) =  b1;
    Q1.at<double>(1, 1) =  a1;
    Q1.at<double>(1, 2) = -d1;
    Q1.at<double>(1, 3) =  c1;

    // Row 3
    Q1.at<double>(2, 0) =  c1;
    Q1.at<double>(2, 1) =  d1;
    Q1.at<double>(2, 2) =  a1;
    Q1.at<double>(2, 3) = -b1;

    // Row 4
    Q1.at<double>(3, 0) =  d1;
    Q1.at<double>(3, 1) = -c1;
    Q1.at<double>(3, 2) =  b1;
    Q1.at<double>(3, 3) =  a1;

    Mat Q2(4, 3, CV_64FC1);

    double w1 = x.at<double>(10, 0);
    double w2 = x.at<double>(11, 0);
    double w3 = x.at<double>(12, 0);

    if (w1 == 0 && w2 == 0 && w3 == 0) {

        // Compute D(q(w*dt))/D(w_old)

        Q2.setTo(Scalar(0));

        Q2.at<double>(1, 0) = 0.5 * dt;
        Q2.at<double>(2, 1) = 0.5 * dt;
        Q2.at<double>(3, 2) = 0.5 * dt;

    } else {

        // Set D(q_new)/D(q_old)

        // Rotation quaternion q(w*dt)
        Mat q = computeQuaternion(x(Rect(0, 10, 1, 3)) * dt);

        double a2 = q.at<double>(0, 0);
        double b2 = q.at<double>(1, 0);
        double c2 = q.at<double>(2, 0);
        double d2 = q.at<double>(3, 0);

        // Row 1
        F.at<double>(3, 3) =  a2;
        F.at<double>(3, 4) = -b2;
        F.at<double>(3, 5) = -c2;
        F.at<double>(3, 6) = -d2;

        // Row 2
        F.at<double>(4, 3) =  b2;
        F.at<double>(4, 4) =  a2;
        F.at<double>(4, 5) =  d2;
        F.at<double>(4, 6) = -c2;

        // Row 3
        F.at<double>(5, 3) =  c2;
        F.at<double>(5, 4) = -d2;
        F.at<double>(5, 5) =  a2;
        F.at<double>(5, 6) =  b2;

        // Row 4
        F.at<double>(6, 3) =  d2;
        F.at<double>(6, 4) =  c2;
        F.at<double>(6, 5) = -b2;
        F.at<double>(6, 6) =  a2;

        // Compute D(q(w*dt))/D(w_old)

        double w11 = w1 * w1;
        double w22 = w2 * w2;
        double w33 = w3 * w3;

        double norm2 = w11 + w22 + w33;
        double norm = sqrt(norm2);

        double c = cos(0.5 * dt * norm);
        double s = sin(0.5 * dt * norm);

        double z = s / norm;

        double dq1dw_ = - 0.5 * dt * z;
        double dq_dw_ = 0.5 * dt * c / norm2 - z / norm2;

        double w12 = w1 * w2;
        double w13 = w1 * w3;
        double w23 = w2 * w3;

        // Row 1
        Q2.at<double>(0, 0) = dq1dw_ * w1;
        Q2.at<double>(0, 1) = dq1dw_ * w2;
        Q2.at<double>(0, 2) = dq1dw_ * w3;

        // Row 2
        Q2.at<double>(1, 0) = dq_dw_ * w11 + z;
        Q2.at<double>(1, 1) = dq_dw_ * w12;
        Q2.at<double>(1, 2) = dq_dw_ * w13;

        // Row 3
        Q2.at<double>(2, 0) = dq_dw_ * w12;
        Q2.at<double>(2, 1) = dq_dw_ * w22 + z;
        Q2.at<double>(2, 2) = dq_dw_ * w23;

        // Row 4
        Q2.at<double>(3, 0) = dq_dw_ * w13;
        Q2.at<double>(3, 1) = dq_dw_ * w23;
        Q2.at<double>(3, 2) = dq_dw_ * w33 + z;
    }

    // Set D(q_new)/D(w_old)

    F(Rect(10, 3, 3, 4)) = Q1 * Q2;

    return F;
}

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
Mat Map::computeProcessNoiseMatrix(double dt, const Mat& F) {

    Mat W0(7, 6, CV_64FC1, Scalar(0));

    // Set D(r_new)/D(V)

    W0.at<double>(0, 0) = dt;
    W0.at<double>(1, 1) = dt;
    W0.at<double>(2, 2) = dt;

    // Set D(q_new)/D(W)

    F(Rect(10, 3, 3, 4)).copyTo(W0(Rect(3, 3, 3, 4)));

    // Compute W * Q * W^t

    Mat Q = A * dt * dt;
    Mat W0_Q = W0 * Q;
    Mat W0_Q_t = W0_Q.t();
    Mat W0_Q_W0 = W0_Q * W0.t();

    Mat N(P.size(), CV_64FC1, Scalar(0));

    W0_Q_W0.copyTo(N(Rect(0, 0, 7, 7)));
    W0_Q.copyTo(N(Rect(7, 0, 6, 7)));
    W0_Q_t.copyTo(N(Rect(0, 7, 7, 6)));
    Q.copyTo(N(Rect(7, 7, 6, 6)));

    return N;
}

/*
 * Returns the Jacobian matrix of the features measurement function with respect
 * to the complete state (r, q, v, w, f1, ..., fn). This matrix is evaluated at
 * the predicted (a priori) state estimate, therefore it should only be called
 * after applyMotionModel since the latter updates the current estimate x. Also,
 * it updates the vector of predicted insight features pixel positions, inviewPos,
 * and populates a vector with the predicted in sight features indices.
 */
Mat Map::computeMeasurementMatrix(vector<int>& inviewIndices) {

    inviewPos.clear();

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

    vector<Point3d> points3D;

    Rect box(Point2i(0, 0), camera.frameSize);

    for (unsigned int i = 0; i < points2D_.size(); i++) {

        if (box.contains(points2D_[i])) {

            inviewIndices.push_back(i);
            inviewPos.push_back(points2D_[i]);

            points3D.push_back(points3D_[i]);
        }
    }

    Mat H(2 * inviewIndices.size(), x.rows, CV_64FC1, Scalar(0));

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
    }

    return H;
}

/*
 * Returns the ellipses where the predicted in sight features should be found with
 * high probability. In particular, the ellipses are constructed so that they are
 * confidence regions at level 0.99.
 *
 * means    Predicted in sight features pixel locations
 * S        Innovation covariance matrix of the predicted in sight features
 */
vector<RotatedRect> Map::computeEllipses(const vector<Point2d>& means, const Mat& S) {

    vector<RotatedRect> ellipses;

    for (unsigned int i = 0; i < means.size(); i++) {

        Mat Si = S(Rect(2*i, 2*i, 2, 2));

        // Compute the eigenvalues and eigenvectors
        Mat eigenval, eigenvec;
        eigen(Si, eigenval, eigenvec);

        // Compute the angle between the largest eigenvector and the x axis
        double angle = atan2(eigenvec.at<double>(0, 1), eigenvec.at<double>(0, 0));

        // Shift the angle from [-pi, pi] to [0, 2pi]
        if (angle < 0)
            angle += PI_DOUBLE;

        // Convert to degrees
        angle *= RAD_TO_DEG;

        // Compute the size of the major and minor axes
        double majorAxis = 6.06970851754 * sqrt(eigenval.at<double>(0));
        double minorAxis = 6.06970851754 * sqrt(eigenval.at<double>(1));

        ellipses.push_back(RotatedRect(means[i], Size(majorAxis, minorAxis), -angle));
    }

    return ellipses;
}

void Map::drawFeatures(Mat& frame,
                       const vector<int>& matchedInviewIndices, const vector<int>& failedInviewIndices,
                       const vector<RotatedRect>& ellipses, bool drawEllipses) {

    if (drawEllipses) {

        // Draw matched features ellipses (red)
        for (unsigned int i = 0; i < matchedInviewIndices.size(); i++) {

            int idx = matchedInviewIndices[i];

            drawEllipse(frame, ellipses[idx], Scalar(0, 0, 255));
        }

        // Draw failed features ellipses (blue)
        for (unsigned int i = 0; i < failedInviewIndices.size(); i++) {

            int idx = failedInviewIndices[i];

            drawEllipse(frame, ellipses[idx], Scalar(255, 0, 0));
        }

    } else {

        // Draw matched features (red)
        for (unsigned int i = 0; i < matchedInviewIndices.size(); i++) {

            int idx = matchedInviewIndices[i];

            drawSquare(frame, inviewPos[idx], 11, Scalar(0, 0, 255));
        }
    }
}

void Map::removeBadFeatures(const vector<int>& failedInviewIndices, const vector<int>& failedIndices) {

    vector<int> badFeaturesVecIndices;  // Indices to remove from features, x and P
    vector<int> badFeaturesPosIndices;  // Indices to remove from inviewPos

    for (unsigned int i = 0; i < failedIndices.size(); i++) {

        int idx = failedIndices[i];
        int j = failedInviewIndices[i];

        double numFails = features[idx].matchingFails;
        double numAttemps = features[idx].matchingAttempts;

        if (numFails / numAttemps > failTolerance) {

            badFeaturesVecIndices.push_back(idx);
            badFeaturesPosIndices.push_back(j);
        }
    }

    if (badFeaturesVecIndices.empty())
        return;

    removeIndices(features, badFeaturesVecIndices);
    removeIndices(inviewPos, badFeaturesPosIndices);

    x = removeRows(x, badFeaturesVecIndices, 13, 3);
    P = removeRowsCols(P, badFeaturesVecIndices, 13, 3);
}

/*
 * Normalizes the system state orientation quaternion and updates the covariance
 * matrix accordingly by first-order error propagation. It is mandatory that the
 * function be called after the EKF update step since the corrected (a posteriori)
 * orientation quaternion might not be a unit quaternion.
 */
void Map::renormalizeQuaternion() {

    Mat q = x(Rect(0, 3, 1, 4));

    double q1 = q.at<double>(0, 0);
    double q2 = q.at<double>(1, 0);
    double q3 = q.at<double>(2, 0);
    double q4 = q.at<double>(3, 0);

    double q11 = q1 * q1;
    double q22 = q2 * q2;
    double q33 = q3 * q3;
    double q44 = q4 * q4;

    double j1 = q22 + q33 + q44;
    double j2 = q11 + q33 + q44;
    double j3 = q11 + q22 + q44;
    double j4 = q11 + q22 + q33;

    double j12 = - q1 * q2;
    double j13 = - q1 * q3;
    double j14 = - q1 * q4;
    double j23 = - q2 * q3;
    double j24 = - q2 * q4;
    double j34 = - q3 * q4;

    Mat J = Mat::eye(P.size(), P.type());

    J.at<double>(3, 3) = j1;
    J.at<double>(3, 4) = j12;
    J.at<double>(3, 5) = j13;
    J.at<double>(3, 6) = j14;

    J.at<double>(4, 3) = j12;
    J.at<double>(4, 4) = j2;
    J.at<double>(4, 5) = j23;
    J.at<double>(4, 6) = j24;

    J.at<double>(5, 3) = j13;
    J.at<double>(5, 4) = j23;
    J.at<double>(5, 5) = j3;
    J.at<double>(5, 6) = j34;

    J.at<double>(6, 3) = j14;
    J.at<double>(6, 4) = j24;
    J.at<double>(6, 5) = j34;
    J.at<double>(6, 6) = j4;

    double qnorm = norm(q);

    J(Rect(3, 3, 4, 4)) /= qnorm * qnorm * qnorm;

    // Normalize quaternion
    normalize(q, q);

    // Update covariance matrix
    P = J * P * J.t();
}