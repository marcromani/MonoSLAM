#ifndef MONOSLAM_CAMERA_H
#define MONOSLAM_CAMERA_H

#include <vector>

#include <opencv2/core.hpp>

namespace monoslam {

class Camera
{
  public:
    cv::Vec3d r;  // Position, in world coordinates
    cv::Vec3d v;  // Linear velocity, in world coordinates
    cv::Vec3d w;  // Angular velocity, in camera coordinates

    cv::Matx<double, 13, 13> P;  // Camera state (r, q_, v, w) covariance matrix

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

    void setOrientation(const cv::Vec4d& q);

    /*
     * Projects 3D points onto the camera frame.
     */
    std::vector<cv::Point2d> projectPoints(const std::vector<cv::Point3d>& points3D) const;

  private:
    cv::Vec4d q_;    // Orientation quaternion (R_ = quaternion::toMatrix(q_))
    cv::Matx33d R_;  // Rotation matrix (from camera basis to world basis)
};

}

#endif
