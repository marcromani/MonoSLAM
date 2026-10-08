#ifndef MONOSLAM_CAMERA_H
#define MONOSLAM_CAMERA_H

#include <vector>

#include <opencv2/core.hpp>

class Camera
{
  public:
    cv::Matx31d r;  // Position, in world coordinates
    cv::Matx41d q;  // Orientation quaternion
    cv::Matx31d v;  // Linear velocity, in world coordinates
    cv::Matx31d w;  // Angular velocity, in camera coordinates

    cv::Mat_<double> P;  // Camera state covariance matrix

    cv::Matx33d R;                  // Rotation matrix (from camera basis to world basis)
    cv::Matx33d K;                  // Intrinsic matrix
    cv::Vec<double, 5> distCoeffs;  // Lens distortion coefficients

    cv::Size frameSize;  // Frame size, in pixels (width, height)

    /*
     * Constructor for a lens-distorted camera model.
     */
    Camera(const cv::Matx33d& K, const cv::Vec<double, 5>& distCoeffs, const cv::Size& frameSize);

    /*
     * Constructor for a pinhole camera model (distCoeffs is filled with zeros).
     */
    Camera(const cv::Matx33d& K, const cv::Size& frameSize);

    /*
     * Warps a rectangular patch (presumed to be the projection of a certain
     * region of a plane in space) from one camera view to another. In particular,
     * a map M is computed such that s2 = M * s1, where s1 is the patch in the
     * first view and s2 is the same patch as would be seen from the second view.
     * A pinhole camera model is assumed (i.e. zero distortion coefficients) so that
     * the computed map be a planar homography. This is always a good approximation
     * provided the planar patch is small and/or the lens distortion is modest.
     *
     * p         Point in the plane, in world coordinates
     * n         Plane normal, in world coordinates
     * view1     Original view
     * patch1    Original patch
     * R1        First pose rotation (world to camera)
     * t1        First pose translation, in world coordinates
     * R2        Second pose rotation (world to camera)
     * t2        Second pose translation, in world coordinates
     * u         First pixel coordinate of the projection of the original patch center
     * v         Second pixel coordinate of the projection of the original patch center
     */
    cv::Mat_<uchar> warpPatch(const cv::Matx31d& p,
                              const cv::Matx31d& n,
                              const cv::Mat_<uchar>& view1,
                              const cv::Rect& patch1,
                              const cv::Matx33d& R1,
                              const cv::Matx31d& t1,
                              const cv::Matx33d& R2,
                              const cv::Matx31d& t2,
                              int& u,
                              int& v) const;

    /*
     * Projects 3D points onto the camera frame.
     *
     * R           Rotation matrix (world to camera)
     * t           Camera position, in world coordinates
     * points3D    Points to be projected, in world coordinates
     * points2D    Point projections, in pixels
     */
    void projectPoints(const cv::Matx33d& R,
                       const cv::Matx31d& t,
                       const std::vector<cv::Point3d>& points3D,
                       std::vector<cv::Point2d>& points2D) const;
};

#endif
