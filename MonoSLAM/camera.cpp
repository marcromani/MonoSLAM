#include "camera.h"

#include "quaternion.h"

monoslam::CameraPose::CameraPose(const cv::Vec3d& r, const cv::Vec4d& q)
  : r_(r)
  , q_(q)
  , R_(quaternion::toMatrix(q).t())
{
    cv::Matx33d R1, R2, R3, R4;
    quaternion::toMatrixDerivatives(q, R1, R2, R3, R4);

    dR_[0] = R1;
    dR_[1] = R2;
    dR_[2] = R3;
    dR_[3] = R4;
}

monoslam::CameraModel::CameraModel(const cv::Matx33d& K,
                                   const cv::Vec<double, 5>& distCoeffs,
                                   const cv::Size& frameSize)
  : K_(K)
  , distCoeffs_(distCoeffs)
  , frameSize_(frameSize)
{
}

monoslam::CameraModel::CameraModel(const cv::Matx33d& K, const cv::Size& frameSize)
  : K_(K)
  , distCoeffs_(cv::Vec<double, 5>::all(0.0))
  , frameSize_(frameSize)
{
}

std::optional<cv::Point2d>
monoslam::CameraModel::project(const CameraPose& pose, const cv::Vec3d& p) const
{
    const cv::Vec3d pCamera = pose.R().t() * (p - pose.r());

    if (pCamera[2] <= 0.0) {
        return std::nullopt;
    }

    return distort(pCamera[0] / pCamera[2], pCamera[1] / pCamera[2]);
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
