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
 * the world frame origin and initial camera pose.
 *
 * frame            Grayscale frame
 * patternSize      Pattern dimensions (see findChessboardCorners)
 * squareSize       Size of a chessboard square, in meters
 * var              Variance of the initial camera state
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

    // Update the map state vector with the camera state vector (zero linear and angular velocity)
    t.copyTo(x(Rect(0, 0, 1, 3)));
    q.copyTo(x(Rect(0, 3, 1, 4)));

    // Update the map covariance matrix with the camera covariance matrix
    Mat var_r = var[0] * Mat::eye(3, 3, CV_64FC1);
    Mat var_q = var[1] * Mat::eye(4, 4, CV_64FC1);
    Mat var_v = var[2] * Mat::eye(3, 3, CV_64FC1);
    Mat var_w = var[3] * Mat::eye(3, 3, CV_64FC1);

    var_r.copyTo(P(Rect(0, 0, 3, 3)));
    var_q.copyTo(P(Rect(3, 3, 4, 4)));
    var_v.copyTo(P(Rect(7, 7, 3, 3)));
    var_w.copyTo(P(Rect(10, 10, 3, 3)));

    // Feature position covariance (position is known with zero uncertainty)
    Mat cov(3, 3, CV_64FC1, Scalar(0));

    // Extract the chessboard outer corners and initialize features
    int idx;

    idx = 0;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], cov, R, t);

    idx = patternSize.width - 1;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], cov, R, t);

    idx = (patternSize.height - 1) * patternSize.width;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], cov, R, t);

    idx = patternSize.width * patternSize.height - 1;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], cov, R, t);

    update();

    return true;
}

void Map::addInitialFeature(const Mat& frame, const Point2f& pos2D, const Point3f& pos3D, const Mat& cov,
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

    // Copy the new feature covariance matrix into the map state covariance matrix
    cov.copyTo(P(Rect(P.cols - 3, P.rows - 3, 3, 3)));

    // Compute the feature patch normal
    double n[] = {0, 0, 1};
    Mat normal(3, 1, CV_64FC1, n);

    features.push_back(Feature(frame,
                               buildSquare(Point2i(pos2D), patchSize),
                               normal,
                               R,
                               t));
}

void Map::update() {

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