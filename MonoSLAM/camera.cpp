#include "camera.hpp"

#include <opencv2/core/types.hpp>

#include "quaternion.hpp"

monoslam::Camera::Camera(const cv::Matx33d& K_, const cv::Size& frameSize_)
  : K(K_)
  , distCoeffs(cv::Vec<double, 5>::all(0.0))
  , frameSize(frameSize_)
{
}

monoslam::Camera::Camera(const cv::Matx33d& K_, const cv::Vec<double, 5>& distCoeffs_, const cv::Size& frameSize_)
  : K(K_)
  , distCoeffs(distCoeffs_)
  , frameSize(frameSize_)
{
}

void
monoslam::Camera::setOrientation(const cv::Vec4d& q)
{
    q_ = q;
    R_ = quaternion::toMatrix(q);
}

std::vector<cv::Point2d>
monoslam::Camera::projectPoints(const std::vector<cv::Point3d>& points3D) const
{
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    const double k1 = distCoeffs[0];
    const double k2 = distCoeffs[1];
    const double p1 = distCoeffs[2];
    const double p2 = distCoeffs[3];
    const double k3 = distCoeffs[4];

    std::vector<cv::Point2d> points2D(points3D.size());

    std::transform(points3D.begin(), points3D.end(), points2D.begin(), [&](const cv::Point3d& pt) {
        const cv::Vec3d pCam = R_.t() * (cv::Vec3d(pt) - r);

        const double zInv = 1.0 / pCam[2];
        const double xn = pCam[0] * zInv;
        const double yn = pCam[1] * zInv;

        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;

        const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        const double tangentialX = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
        const double tangentialY = 2.0 * p2 * xn * yn + p1 * (r2 + 2.0 * yn * yn);

        return cv::Point2d(fx * (xn * radial + tangentialX) + cx, fy * (yn * radial + tangentialY) + cy);
    });

    return points2D;
}
