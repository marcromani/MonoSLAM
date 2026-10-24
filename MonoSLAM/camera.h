#ifndef MONOSLAM_CAMERA_H
#define MONOSLAM_CAMERA_H

#include <array>
#include <optional>

#include <opencv2/core.hpp>

namespace monoslam {

class CameraPose
{
  public:
    CameraPose(const cv::Vec3d& r, const cv::Vec4d& q);

    const cv::Vec3d& r() const { return r_; }
    const cv::Vec4d& q() const { return q_; }
    const cv::Matx33d& R() const { return R_; }
    const std::array<cv::Matx33d, 4>& dR() const { return dR_; }

  private:
    cv::Vec3d r_;                    // Position, in world coordinates
    cv::Vec4d q_;                    // Orientation quaternion
    cv::Matx33d R_;                  // Rotation matrix, R(q)^t (from world basis to camera basis)
    std::array<cv::Matx33d, 4> dR_;  // Derivatives of R_ w.r.t. each quaternion component
};

struct Projection
{
    cv::Vec2d pixel;
    cv::Matx<double, 2, 3> Jr;  // Jacobian of the pixel w.r.t. the camera position
    cv::Matx<double, 2, 4> Jq;  // Jacobian of the pixel w.r.t. the camera quaternion
    cv::Matx<double, 2, 3> Jp;  // Jacobian of the pixel w.r.t. the world point
};

class CameraModel
{
  public:
    /*
     * Constructor for a lens-distorted camera model.
     */
    CameraModel(const cv::Matx33d& K, const cv::Vec<double, 5>& distCoeffs, const cv::Size& frameSize);

    /*
     * Constructor for a pinhole camera model (distCoeffs is filled with zeros).
     */
    CameraModel(const cv::Matx33d& K, const cv::Size& frameSize);

    const cv::Matx33d& K() const { return K_; }
    const cv::Vec<double, 5>& distCoeffs() const { return distCoeffs_; }
    const cv::Size& frameSize() const { return frameSize_; }

    /*
     * Maps the components of a 3D point from the affine world frame to camera pixel coordinates.
     * If the point is not visible std::nullopt is returned.
     */
    std::optional<cv::Point2d> project(const CameraPose& pose, const cv::Vec3d& p) const;

    /*
     * Maps the components of a 3D point from the affine world frame to camera pixel coordinates.
     * Additionally, it computes the Jacobian of this projection w.r.t. the camera pose and the
     * 3D point itself. If the point is not visible std::nullopt is returned.
     */
    std::optional<Projection> projectWithJacobian(const CameraPose& pose, const cv::Vec3d& p) const;

    bool inView(const cv::Point2d& pixel) const;

  private:
    cv::Point2d distort(double x, double y) const;

    cv::Matx33d K_;                  // Intrinsic matrix
    cv::Vec<double, 5> distCoeffs_;  // Lens distortion coefficients
    cv::Size frameSize_;             // Frame size, in pixels (width, height)
};

}

#endif
