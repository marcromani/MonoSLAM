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
    P = Mat::eye(13, 13, CV_64FC1) * Mat(var);

    // Extract the chessboard outer corners and add initial features
    int idx;

    idx = 0;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = patternSize.width - 1;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = (patternSize.height - 1) * patternSize.width;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    idx = patternSize.width * patternSize.height - 1;
    addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t);

    return true;
}

bool Map::trackNewCandidates(const cv::Mat& frame) {

    // If there are enough features there is no need to detect new ones
    if (features.size() >= minFeatureDensity)
        return false;

    // The same applies if there are few features but candidates are already being tracked
    if (!candidates.empty())
        return false;

    vector<Point2f> corners = findCorners(frame);

    // TODO: add pre-initialized features

    return true;
}

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
void Map::addInitialFeature(const Mat& frame, const Point2f& pos2D, const Point3f& pos3D,
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

    features.push_back(Feature(frame,
                               buildSquare(Point2i(pos2D), patchSize),
                               normal,
                               R,
                               t));
}

/*
 * Updates the camera and features data with the map state vector and covariance matrix
 * data. Only shallow copies are performed. This function should be called whenever the
 * current state of some (or all) of the system parts is required.
 */
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

vector<Point2f> Mat::findCorners(const Mat& frame) {

    vector<Point2f> corners;

    int maxCorners = maxFeatureDensity - features.size();
    double qualityLevel = 0.2;
    double minDistance = 60;

    Mat mask(frame.size(), CV_8UC1, Scalar(0));

    // Set a margin around the mask to enclose the detected corners inside
    // a rectangle and avoid premature corner loss due to camera movement
    int pad = 30;
    mask(Rect(pad, pad, mask.cols - 2 * pad, mask.rows - 2 * pad)).setTo(Scalar(255));

    // Compute the additional length that a feature patch must have
    // so that new detected corners are the same minimum distance apart
    // from already initialized features than from themselves
    int len = 2 * minDistance - patchSize;

    for (unsigned int i = 0; i < features.size(); i++) {

        Rect roi = features[i].roi;

        roi.x = max(0, roi.x - len / 2);
        roi.y = max(0, roi.y - len / 2);

        roi.width += len;
        roi.height += len;

        if (roi.x + roi.width > frame.cols)
            roi.width = frame.cols - roi.x;
        if (roi.y + roi.height > frame.rows)
            roi.height = frame.rows - roi.y;

        // Update the mask to reject the region around the ith feature
        mask(roi).setTo(Scalar(0));
    }

    goodFeaturesToTrack(frame, corners, maxCorners, qualityLevel, minDistance, mask);

    return corners;
}