#ifndef CAMERA_H
#define CAMERA_H

#include <vector>

#include "opencv2/core/core.hpp"

class Camera {

public:

    cv::Mat r;              // Position, in world coordinates
    cv::Mat q;              // Orientation quaternion
    cv::Mat v;              // Linear velocity, in world coordinates
    cv::Mat w;              // Angular velocity, in camera coordinates

    cv::Mat P;              // Camera state covariance matrix

    cv::Mat R;              // Rotation matrix (from camera basis to world basis)
    cv::Mat K;              // Intrinsic matrix
    cv::Mat distCoeffs;     // Lens distortion coefficients

    cv::Mat distortionMap;  // Mapping from distorted pixels to undistorted pixels

    cv::Size frameSize;     // Frame size, in pixels

    /*
     * Constructor for a lens distortion camera model.
     */
    Camera(const cv::Mat& K, const cv::Mat& distCoeffs, const cv::Size& frameSize);

    /*
     * Constructor for a pinhole camera model (distCoeffs is left empty by default).
     */
    Camera(const cv::Mat& K, const cv::Size& frameSize);

    /*
     * Returns the distorted version of a camera image, for which no lens distortion
     * is applied yet (only the pinhole model). It is the inverse operation of undistort.
     */
    cv::Mat distort(const cv::Mat& image);

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
    cv::Mat warpPatch(const cv::Mat& p, const cv::Mat& n, const cv::Mat& view1, const cv::Rect& patch1,
                      const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, const cv::Mat& t2);

    cv::Mat warpPatch2(const cv::Mat& p, const cv::Mat& n, const cv::Mat& view1, const cv::Rect& patch1,
                       const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, const cv::Mat& t2,
                       int& u, int& v);

    void projectPoints(const cv::Mat& R, const cv::Mat& t, const std::vector<cv::Point3d>& points3D,
                       std::vector<cv::Point2d>& points2D);
};

#endif