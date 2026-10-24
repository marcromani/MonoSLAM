#include "opencv2/imgproc/imgproc.hpp"

#include "camera.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

/*
 * Constructor for a lens distortion camera model.
 */
Camera::Camera(const Mat& K_, const Mat& distCoeffs_, const Size& frameSize_) {

    K_.copyTo(K);
    frameSize = frameSize_;

    distCoeffs_.copyTo(distCoeffs);
}

/*
 * Constructor for a pinhole camera model (distCoeffs is filled with zeros).
 */
Camera::Camera(const Mat& K_, const Size& frameSize_) {

    K_.copyTo(K);
    frameSize = frameSize_;

    distCoeffs = Mat::zeros(1, 5, CV_64FC1);
}

/*
 * Warps a rectangular patch (presumed to be the projection of a certain region of a plane in space)
 * from one camera view to another. In particular, a map M is computed such that s2 = M * s1, where
 * s1 is the patch in the first view and s2 is the appearance of the same patch as would be seen from
 * the second view. A pinhole camera model is assumed (i.e. zero distortion coefficients) so that the
 * computed map be a planar homography. This is always a good approximation provided that the planar
 * patch is small or that the lens distortion is modest.
 *
 * p            Point in the plane, in world coordinates
 * n            Plane normal, in world coordinates
 * view1        Original view
 * patch1       Original patch
 * R1           First pose rotation (world to camera)
 * t1           First pose translation, in world coordinates
 * R2           Second pose rotation (world to camera)
 * t2           Second pose translation, in world coordinates
 * u            First pixel coordinate of the projection of the original patch center
 * v            Second pixel coordinate of the projection of the original patch center
 * mask         Warped template mask, it specifies the pixels that belong to the original patch
 */
Mat Camera::warpPatch(const Mat& p, const Mat& n, const Mat& view1, const Rect& patch1,
                      const Mat& R1, const Mat& t1, const Mat& R2, const Mat& t2,
                      int& u, int& v, Mat& mask) {

    // Black background
    Mat black(view1.size(), view1.type(), Scalar::all(0));

    // Copy the original patch into the black background
    view1(patch1).copyTo(black(patch1));

    // Compute the homography matrix between the first and second views
    Mat H = computeHomography(p, n, K, R1, t1, R2, t2);

    // Apply the homography
    Mat warped;
    warpPerspective(black, warped, H, view1.size());

    // Set every non-black pixel of the warped image to white
    threshold(warped, mask, 0, 255, THRESH_BINARY);

    // Use the binary mask to crop the warped image so that it only contains the patch
    Mat points;
    findNonZero(mask, points);
    Rect patchBox = boundingRect(points);

    // Compute the projection of the center of the patch
    int x = patch1.x + patch1.width / 2;
    int y = patch1.y + patch1.height / 2;

    Mat centerProj = H * (Mat_<double>(3, 1) << x, y, 1);
    centerProj /= centerProj.at<double>(2, 0);

    u = centerProj.at<double>(0, 0) - patchBox.x;
    v = centerProj.at<double>(1, 0) - patchBox.y;

    mask = mask(patchBox);

    return warped(patchBox);
}

/*
 * Projects 3D points onto the camera frame.
 *
 * R            Rotation matrix (world to camera)
 * t            Camera position, in world coordinates
 * points3D     Points to be projected, in world coordinates
 * points2D     Points projections, in pixels
 */
void Camera::projectPoints(const Mat& R, const Mat& t, const vector<Point3d>& points3D,
                           vector<Point2d>& points2D) {

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(0, 1);
    double p1 = distCoeffs.at<double>(0, 2);
    double p2 = distCoeffs.at<double>(0, 3);
    double k3 = distCoeffs.at<double>(0, 4);

    points2D.resize(points3D.size());

    for (unsigned int i = 0; i < points3D.size(); i++) {

        Mat pCam = R * (Mat(points3D[i]) - t);

        double xn = pCam.at<double>(0, 0) / pCam.at<double>(2, 0);
        double yn = pCam.at<double>(1, 0) / pCam.at<double>(2, 0);

        double r2 = xn*xn + yn*yn;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double u = xn * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*xn*yn + p2*(r2 + 2*xn*xn);
        double v = yn * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p2*xn*yn + p1*(r2 + 2*yn*yn);

        u = fx*u + cx;
        v = fy*v + cy;

        points2D[i] = Point2d(u, v);
    }
}