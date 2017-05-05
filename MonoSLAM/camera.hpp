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

    cv::Size frameSize;     // Frame size, in pixels

    /*
     * Constructor for a lens distortion camera model.
     */
    Camera(const cv::Mat& K, const cv::Mat& distCoeffs, const cv::Size& frameSize);

    /*
     * Constructor for a pinhole camera model (distCoeffs is filled with zeros).
     */
    Camera(const cv::Mat& K, const cv::Size& frameSize);

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
     */
    cv::Mat warpPatch(const cv::Mat& p, const cv::Mat& n, const cv::Mat& view1, const cv::Rect& patch1,
                      const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, const cv::Mat& t2,
                      int& u, int& v);

    /*
     * Projects 3D points onto the camera frame.
     *
     * R            Rotation matrix (world to camera)
     * t            Camera position, in world coordinates
     * points3D     Points to be projected, in world coordinates
     * points2D     Points projections, in pixels
     */
    void projectPoints(const cv::Mat& R, const cv::Mat& t, const std::vector<cv::Point3d>& points3D,
                       std::vector<cv::Point2d>& points2D);
};

#endif