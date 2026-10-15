#include "camera.h"

#include <optional>

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

    cv::Point2d pixel = distort(pCamera[0] / pCamera[2], pCamera[1] / pCamera[2]);

    if (!inView(pixel)) {
        return std::nullopt;
    }

    return pixel;
}

std::optional<monoslam::Projection>
monoslam::CameraModel::projectWithJacobian(const CameraPose& pose, const cv::Vec3d& p) const
{
    const cv::Matx33d R = pose.R();

    const cv::Vec3d t = p - pose.r();
    const cv::Vec3d pCamera = R.t() * t;

    if (pCamera[2] <= 0.0) {
        return std::nullopt;
    }

    const cv::Point2d pixel = distort(pCamera[0] / pCamera[2], pCamera[1] / pCamera[2]);

    if (!inView(pixel)) {
        return std::nullopt;
    }

    const double fx = K_(0, 0);
    const double fy = K_(1, 1);

    const double k1 = distCoeffs_[0];
    const double k2 = distCoeffs_[1];
    const double p1 = distCoeffs_[2];
    const double p2 = distCoeffs_[3];
    const double k3 = distCoeffs_[4];

    const cv::Matx33d& R1 = pose.dR()[0];
    const cv::Matx33d& R2 = pose.dR()[1];
    const cv::Matx33d& R3 = pose.dR()[2];
    const cv::Matx33d& R4 = pose.dR()[3];

    const double xCam = pCamera[0];
    const double yCam = pCamera[1];
    const double zCam = pCamera[2];

    const double zCam2 = zCam * zCam;

    const cv::Vec3d pBar1 = R1 * t;
    const cv::Vec3d pBar2 = R2 * t;
    const cv::Vec3d pBar3 = R3 * t;
    const cv::Vec3d pBar4 = R4 * t;

    // D{xn / r}: derivatives of the normalized camera coordinate xn with respect to camera position
    const double dxndr1 = (R(2, 0) * xCam - R(0, 0) * zCam) / zCam2;
    const double dxndr2 = (R(2, 1) * xCam - R(0, 1) * zCam) / zCam2;
    const double dxndr3 = (R(2, 2) * xCam - R(0, 2) * zCam) / zCam2;

    // D{xn / q}: derivatives of the normalized camera coordinate xn with respect to camera orientation
    const double dxndq1 = (pBar1[0] * zCam - pBar1[2] * xCam) / zCam2;
    const double dxndq2 = (pBar2[0] * zCam - pBar2[2] * xCam) / zCam2;
    const double dxndq3 = (pBar3[0] * zCam - pBar3[2] * xCam) / zCam2;
    const double dxndq4 = (pBar4[0] * zCam - pBar4[2] * xCam) / zCam2;

    // D{yn / r}: derivatives of the normalized camera coordinate yn with respect to camera position
    const double dyndr1 = (R(2, 0) * yCam - R(1, 0) * zCam) / zCam2;
    const double dyndr2 = (R(2, 1) * yCam - R(1, 1) * zCam) / zCam2;
    const double dyndr3 = (R(2, 2) * yCam - R(1, 2) * zCam) / zCam2;

    // D{yn / q}: derivatives of the normalized camera coordinate yn with respect to camera orientation
    const double dyndq1 = (pBar1[1] * zCam - pBar1[2] * yCam) / zCam2;
    const double dyndq2 = (pBar2[1] * zCam - pBar2[2] * yCam) / zCam2;
    const double dyndq3 = (pBar3[1] * zCam - pBar3[2] * yCam) / zCam2;
    const double dyndq4 = (pBar4[1] * zCam - pBar4[2] * yCam) / zCam2;

    const double xn = xCam / zCam;
    const double yn = yCam / zCam;

    // D{r2 / r}: derivatives of r^2 = xn^2 + yn^2 with respect to camera position
    const double dr2dr1 = 2.0 * (xn * dxndr1 + yn * dyndr1);
    const double dr2dr2 = 2.0 * (xn * dxndr2 + yn * dyndr2);
    const double dr2dr3 = 2.0 * (xn * dxndr3 + yn * dyndr3);

    // D{r2 / q}: derivatives of r^2 = xn^2 + yn^2 with respect to camera orientation
    const double dr2dq1 = 2.0 * (xn * dxndq1 + yn * dyndq1);
    const double dr2dq2 = 2.0 * (xn * dxndq2 + yn * dyndq2);
    const double dr2dq3 = 2.0 * (xn * dxndq3 + yn * dyndq3);
    const double dr2dq4 = 2.0 * (xn * dxndq4 + yn * dyndq4);

    const double r2 = xn * xn + yn * yn;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double a = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double b = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

    const double ddr1 = 2 * (xn * dyndr1 + yn * dxndr1);
    const double ddr2 = 2 * (xn * dyndr2 + yn * dxndr2);
    const double ddr3 = 2 * (xn * dyndr3 + yn * dxndr3);

    const double ddq1 = 2 * (xn * dyndq1 + yn * dxndq1);
    const double ddq2 = 2 * (xn * dyndq2 + yn * dxndq2);
    const double ddq3 = 2 * (xn * dyndq3 + yn * dxndq3);
    const double ddq4 = 2 * (xn * dyndq4 + yn * dxndq4);

    // D{u / r}
    const double dudr1 = fx * (dxndr1 * a + xn * dr2dr1 * b + p1 * ddr1 + p2 * (dr2dr1 + 4 * xn * dxndr1));
    const double dudr2 = fx * (dxndr2 * a + xn * dr2dr2 * b + p1 * ddr2 + p2 * (dr2dr2 + 4 * xn * dxndr2));
    const double dudr3 = fx * (dxndr3 * a + xn * dr2dr3 * b + p1 * ddr3 + p2 * (dr2dr3 + 4 * xn * dxndr3));

    // D{u / x}
    const double dudx1 = -dudr1;
    const double dudx2 = -dudr2;
    const double dudx3 = -dudr3;

    // D{u / q}
    const double dudq1 = fx * (dxndq1 * a + xn * dr2dq1 * b + p1 * ddq1 + p2 * (dr2dq1 + 4 * xn * dxndq1));
    const double dudq2 = fx * (dxndq2 * a + xn * dr2dq2 * b + p1 * ddq2 + p2 * (dr2dq2 + 4 * xn * dxndq2));
    const double dudq3 = fx * (dxndq3 * a + xn * dr2dq3 * b + p1 * ddq3 + p2 * (dr2dq3 + 4 * xn * dxndq3));
    const double dudq4 = fx * (dxndq4 * a + xn * dr2dq4 * b + p1 * ddq4 + p2 * (dr2dq4 + 4 * xn * dxndq4));

    // D{v / r}
    const double dvdr1 = fy * (dyndr1 * a + yn * dr2dr1 * b + p2 * ddr1 + p1 * (dr2dr1 + 4 * yn * dyndr1));
    const double dvdr2 = fy * (dyndr2 * a + yn * dr2dr2 * b + p2 * ddr2 + p1 * (dr2dr2 + 4 * yn * dyndr2));
    const double dvdr3 = fy * (dyndr3 * a + yn * dr2dr3 * b + p2 * ddr3 + p1 * (dr2dr3 + 4 * yn * dyndr3));

    // D{v / x}
    const double dvdx1 = -dvdr1;
    const double dvdx2 = -dvdr2;
    const double dvdx3 = -dvdr3;

    // D{v / q}
    const double dvdq1 = fy * (dyndq1 * a + yn * dr2dq1 * b + p2 * ddq1 + p1 * (dr2dq1 + 4 * yn * dyndq1));
    const double dvdq2 = fy * (dyndq2 * a + yn * dr2dq2 * b + p2 * ddq2 + p1 * (dr2dq2 + 4 * yn * dyndq2));
    const double dvdq3 = fy * (dyndq3 * a + yn * dr2dq3 * b + p2 * ddq3 + p1 * (dr2dq3 + 4 * yn * dyndq3));
    const double dvdq4 = fy * (dyndq4 * a + yn * dr2dq4 * b + p2 * ddq4 + p1 * (dr2dq4 + 4 * yn * dyndq4));

    Projection projection;

    projection.pixel = pixel;
    projection.Jr = { dudr1, dudr2, dudr3, dvdr1, dvdr2, dvdr3 };
    projection.Jq = { dudq1, dudq2, dudq3, dudq4, dvdq1, dvdq2, dvdq3, dvdq4 };
    projection.Jp = { dudx1, dudx2, dudx3, dvdx1, dvdx2, dvdx3 };

    return projection;
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
