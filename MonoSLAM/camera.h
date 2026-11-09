#ifndef MONOSLAM_CAMERA_H
#define MONOSLAM_CAMERA_H

#include <optional>
#include <string>

#include <opencv2/core.hpp>

namespace monoslam {

/*
 * Returns the rotation matrix of a rotation vector, the exponential map of SO(3): the rotation by |v| radians about v.
 */
cv::Matx33d
rotation(const cv::Vec3d& v);

/*
 * Returns the matrix of the cross product with v, [v]x u = v x u.
 */
cv::Matx33d
skew(const cv::Vec3d& v);

/*
 * The pose of a camera, with its rotation both ways, since both are needed many times per frame. A point with camera
 * coordinates p_c has world coordinates R p_c + r.
 */
class CameraPose
{
  public:
    /*
     * r    Position, in world coordinates
     * R    Orientation, the rotation from camera to world coordinates
     */
    CameraPose(const cv::Vec3d& r, const cv::Matx33d& R);

    const cv::Vec3d& r() const { return r_; }
    const cv::Matx33d& R() const { return R_; }
    const cv::Matx33d& Rcw() const { return Rcw_; }

  private:
    cv::Vec3d r_;      // Position, in world coordinates
    cv::Matx33d R_;    // Rotation from camera to world coordinates
    cv::Matx33d Rcw_;  // Rotation from world to camera coordinates, R^T
};

class CameraModel
{
  public:
    /*
     * K             Intrinsic matrix
     * distCoeffs    Lens distortion coefficients (k1, k2, p1, p2, k3), as OpenCV's
     * frameSize     Frame size, in pixels
     */
    CameraModel(const cv::Matx33d& K, const cv::Vec<double, 5>& distCoeffs, const cv::Size& frameSize);

    /*
     * Reads a camera model from a calibration file, as written by the calibration program. Returns nothing if the file
     * cannot be read or misses a value.
     *
     * filename    Calibration file
     */
    static std::optional<CameraModel> load(const std::string& filename);

    const cv::Matx33d& K() const { return K_; }
    const cv::Vec<double, 5>& distCoeffs() const { return distCoeffs_; }
    const cv::Size& frameSize() const { return frameSize_; }

    /*
     * Returns the pixel of a point in world coordinates, seen from a pose, or nothing if it is behind the camera or out
     * of the frame.
     */
    std::optional<cv::Point2d> project(const CameraPose& pose, const cv::Vec3d& p) const;

    /*
     * Projects a point given in camera coordinates, and computes the Jacobian of its pixel with respect to it, lens
     * distortion included. The point can be scaled by any positive factor, the pixel only depends on its direction.
     * Returns false if the point is behind the camera or its pixel is out of the frame.
     *
     * p        Point, in camera coordinates
     * pixel    Its pixel
     * J        Jacobian of the pixel with respect to p
     */
    bool projectCameraPoint(const cv::Vec3d& p, cv::Point2d& pixel, cv::Matx23d& J) const;

    /*
     * Checks whether the pixel is within the frame boundaries or not.
     */
    bool inView(const cv::Point2d& pixel) const;

    /*
     * Maps normalized camera coordinates to pixel coordinates, including lens distortion.
     */
    cv::Point2d distort(double x, double y) const;

    /*
     * Returns the Jacobian of distort with respect to the normalized camera coordinates.
     */
    cv::Matx22d distortionJacobian(double x, double y) const;

    /*
     * Returns the direction of the 3D ray associated to a pixel, in camera coordinates and with unit depth, that is,
     * (x, y, 1) with x and y its normalized camera coordinates.
     */
    cv::Vec3d unproject(const cv::Point2d& pixel) const;

  private:
    cv::Matx33d K_;                  // Intrinsic matrix
    cv::Vec<double, 5> distCoeffs_;  // Lens distortion coefficients
    cv::Size frameSize_;             // Frame size, in pixels (width, height)
};

}

#endif
