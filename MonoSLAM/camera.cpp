#include "opencv2/imgproc/imgproc.hpp"

#include "camera.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;

/*
 * Constructor for a lens distortion camera model.
 */
Camera::Camera(const Mat& r_, const Mat& q_, const Mat& v_, const Mat& w_, const Mat& P_,
               const Mat& K_, const Mat& distCoeffs_, const Size& frameSize_) {

    r_.copyTo(r);
    q_.copyTo(q);
    v_.copyTo(v);
    w_.copyTo(w);
    P_.copyTo(P);
    K_.copyTo(K);
    frameSize = frameSize_;

    if (!distCoeffs_.empty()) {

        distCoeffs_.copyTo(distCoeffs);
        distortionMap = buildDistortionMap(frameSize, K, distCoeffs);
    }

    R = getRotationMatrix(q);
}

/*
 * Constructor for a pinhole camera model (distCoeffs is left empty by default).
 */
Camera::Camera(const Mat& r_, const Mat& q_, const Mat& v_, const Mat& w_, const Mat& P_,
               const Mat& K_, const Size& frameSize_) {

    r_.copyTo(r);
    q_.copyTo(q);
    v_.copyTo(v);
    w_.copyTo(w);
    P_.copyTo(P);
    K_.copyTo(K);
    frameSize = frameSize_;

    R = getRotationMatrix(q);
}

/*
 * Returns the distorted version of a camera image, for which no lens distortion
 * is applied yet (only the pinhole model). It is the inverse operation of undistort.
 */
Mat Camera::distort(const Mat& image) {

    if (distCoeffs.empty())
        return image.clone();

    Mat distorted, empty;
    remap(image, distorted, distortionMap, empty, INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    return distorted;
}

/*
 * Warps a rectangular patch (presumed to be the projection of a certain region of a plane in space)
 * from one camera view to another. In particular, a map M is computed such that s2 = M * s1, where
 * s1 is the patch in the first view and s2 is the appearance of the same patch as would be seen from
 * the second view. Pixels which do not belong to the patch are set to 0. Note that, given the way
 * the image is undistorted and distorted back, some portions of the warped patch might not be
 * recoverable and appear black (especially around the corners), if the warped undistorted patch
 * exceeds the image boundary. This will not occur with a pinhole camera model.
 *
 * p            Point in the plane, in world coordinates
 * n            Plane normal, in world coordinates
 * view1        Original view
 * patch1       Original patch
 * R1           First pose rotation (world to camera)
 * t1           First pose translation, in world coordinates
 * R2           Second pose rotation (world to camera)
 * t2           Second pose translation, in world coordinates
 */
Mat Camera::warpPatch(const Mat& p, const Mat& n, const Mat& view1, const Rect& patch1,
                      const Mat& R1, const Mat& t1, const Mat& R2, const Mat& t2) {

    // Black background
    Mat black(view1.rows, view1.cols, view1.type(), Scalar(0));

    // Copy the original patch into the black background
    view1(patch1).copyTo(black(patch1));

    // Undistort the original patch
    Mat undistorted1;
    undistort(black, undistorted1, K, distCoeffs);

    // Compute the homography matrix between the first and second views
    Mat H = computeHomography(p, n, K, R1, t1, R2, t2);

    // Apply the homography
    Mat warped;
    warpPerspective(undistorted1, warped, H, Size(view1.cols, view1.rows));

    // Distort back the warped image
    Mat distorted2 = distort(warped);

    return distorted2;
}