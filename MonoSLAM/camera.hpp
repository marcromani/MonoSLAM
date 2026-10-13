#ifndef MONOSLAM_CAMERA_H
#define MONOSLAM_CAMERA_H

#include <vector>

#include <opencv2/core.hpp>

namespace monoslam {

class Camera
{
  public:
    cv::Vec3d r;  // Position, in world coordinates
    cv::Vec4d q;  // Orientation quaternion
    cv::Vec3d v;  // Linear velocity, in world coordinates
    cv::Vec3d w;  // Angular velocity, in camera coordinates

    cv::Mat1d P;  // Camera state (r, q, v, w) covariance matrix

    cv::Matx33d R;                  // Rotation matrix (from camera basis to world basis)
    cv::Matx33d K;                  // Intrinsic matrix
    cv::Vec<double, 5> distCoeffs;  // Lens distortion coefficients

    cv::Size frameSize;  // Frame size, in pixels (width, height)

    /*
     * Constructor for a pinhole camera model (distCoeffs is filled with zeros).
     */
    Camera(const cv::Matx33d& K, const cv::Size& frameSize);

    /*
     * Constructor for a lens-distorted camera model.
     */
    Camera(const cv::Matx33d& K, const cv::Vec<double, 5>& distCoeffs, const cv::Size& frameSize);

    /*
     * Projects 3D points onto the camera frame.
     *
     * R           Rotation matrix (world to camera)
     * t           Camera position, in world coordinates
     * points3D    Points to be projected, in world coordinates
     * points2D    Point projections, in pixels
     */
    void projectPoints(const cv::Matx33d& R,
                       const cv::Vec3d& t,
                       const std::vector<cv::Point3d>& points3D,
                       std::vector<cv::Point2d>& points2D) const;
};

}

#endif
