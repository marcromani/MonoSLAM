#include "camera.h"

#include <optional>
#include <vector>

#include <opencv2/core/persistence.hpp>
#include <opencv2/geometry.hpp>

cv::Matx33d
monoslam::rotation(const cv::Vec3d& v)
{
    cv::Matx33d R;
    cv::Rodrigues(v, R);

    return R;
}

cv::Matx33d
monoslam::skew(const cv::Vec3d& v)
{
    return cv::Matx33d(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0);
}

monoslam::CameraPose::CameraPose(const cv::Vec3d& r, const cv::Matx33d& R)
  : r_(r)
  , R_(R)
  , Rcw_(R.t())
{
}

monoslam::CameraModel::CameraModel(const cv::Matx33d& K,
                                   const cv::Vec<double, 5>& distCoeffs,
                                   const cv::Size& frameSize)
  : K_(K)
  , distCoeffs_(distCoeffs)
  , frameSize_(frameSize)
{
}

std::optional<cv::Point2d>
monoslam::CameraModel::project(const CameraPose& pose, const cv::Vec3d& p) const
{
    const cv::Vec3d pCamera = pose.Rcw() * (p - pose.r());

    if (pCamera[2] <= 0.0) {
        return std::nullopt;
    }

    cv::Point2d pixel = distort(pCamera[0] / pCamera[2], pCamera[1] / pCamera[2]);

    if (!inView(pixel)) {
        return std::nullopt;
    }

    return pixel;
}

bool
monoslam::CameraModel::projectCameraPoint(const cv::Vec3d& p, cv::Point2d& pixel, cv::Matx23d& J) const
{
    if (p[2] <= 0.0) {
        return false;
    }

    const double x = p[0] / p[2];
    const double y = p[1] / p[2];

    pixel = distort(x, y);

    if (!inView(pixel)) {
        return false;
    }

    // Normalized coordinates with respect to the point
    const cv::Matx23d normalization(1.0 / p[2], 0.0, -x / p[2], 0.0, 1.0 / p[2], -y / p[2]);

    J = distortionJacobian(x, y) * normalization;

    return true;
}

cv::Matx22d
monoslam::CameraModel::distortionJacobian(double x, double y) const
{
    const double k1 = distCoeffs_[0];
    const double k2 = distCoeffs_[1];
    const double p1 = distCoeffs_[2];
    const double p2 = distCoeffs_[3];
    const double k3 = distCoeffs_[4];

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;

    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2;
    const double radialDerivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;  // Of radial with respect to r2

    // Derivatives of the distorted normalized coordinates, see distort
    const double dxdx = radial + 2.0 * x * x * radialDerivative + 2.0 * p1 * y + 6.0 * p2 * x;
    const double dxdy = 2.0 * x * y * radialDerivative + 2.0 * p1 * x + 2.0 * p2 * y;
    const double dydx = 2.0 * x * y * radialDerivative + 2.0 * p2 * y + 2.0 * p1 * x;
    const double dydy = radial + 2.0 * y * y * radialDerivative + 2.0 * p2 * x + 6.0 * p1 * y;

    return cv::Matx22d(K_(0, 0) * dxdx, K_(0, 0) * dxdy, K_(1, 1) * dydx, K_(1, 1) * dydy);
}

bool
monoslam::CameraModel::inView(const cv::Point2d& pixel) const
{
    return 0.0 <= pixel.x && pixel.x < frameSize_.width && 0.0 <= pixel.y && pixel.y < frameSize_.height;
}

cv::Point2d
monoslam::CameraModel::distort(double x, double y) const
{
    const double fx = K_(0, 0);
    const double fy = K_(1, 1);
    const double cx = K_(0, 2);
    const double cy = K_(1, 2);

    const double k1 = distCoeffs_[0];
    const double k2 = distCoeffs_[1];
    const double p1 = distCoeffs_[2];
    const double p2 = distCoeffs_[3];
    const double k3 = distCoeffs_[4];

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double tangentialX = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double tangentialY = 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y);

    return cv::Point2d(fx * (x * radial + tangentialX) + cx, fy * (y * radial + tangentialY) + cy);
}

cv::Vec3d
monoslam::CameraModel::unproject(const cv::Point2d& pixel) const
{
    const std::vector<cv::Point2d> distorted = { pixel };

    std::vector<cv::Point2d> normalized;
    cv::undistortPoints(distorted, normalized, K_, distCoeffs_);

    return cv::Vec3d(normalized[0].x, normalized[0].y, 1.0);
}

std::optional<monoslam::CameraModel>
monoslam::CameraModel::load(const std::string& filename)
{
    cv::FileStorage file(filename, cv::FileStorage::READ);

    if (!file.isOpened()) {
        return std::nullopt;
    }

    // Intrinsics, then distortion coefficients
    const char* names[] = { "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3" };

    double values[9];

    for (int i = 0; i < 9; ++i) {
        const cv::FileNode node = file[names[i]];

        if (node.empty()) {
            return std::nullopt;
        }

        node >> values[i];
    }

    const cv::FileNode sizeNode = file["Frame size"];

    if (sizeNode.empty()) {
        return std::nullopt;
    }

    cv::Size frameSize;
    sizeNode >> frameSize;

    const cv::Matx33d K(values[0], 0.0, values[2], 0.0, values[1], values[3], 0.0, 0.0, 1.0);
    const cv::Vec<double, 5> distCoeffs(values[4], values[5], values[6], values[7], values[8]);

    return CameraModel(K, distCoeffs, frameSize);
}
