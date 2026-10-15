#include "system_observation_model.h"

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "quaternion.h"

monoslam::SystemObservationModel::SystemObservationModel(const CameraModel& cameraModel)
  : cameraModel_(cameraModel)
{
}

monoslam::filter::Prediction
monoslam::SystemObservationModel::eval(const cv::Mat1d& x) const
{
}

cv::Mat1d
monoslam::SystemObservationModel::h(const cv::Mat1d& x) const
{
    // // Compute the camera rotation (from world basis to camera basis)
    // const cv::Matx33d R = quaternion::toMatrix(x(cv::Rect(0, 3, 1, 4))).t();

    // // Get the camera position (in world coordinates)
    // const cv::Mat1d t = x(cv::Rect(0, 0, 1, 3));

    const int numFeatures = (x.rows - 13) / 3;

    std::vector<cv::Vec3d> vecs3D(static_cast<size_t>(numFeatures));

    // Create a vector of feature positions
    for (int i = 0; i < numFeatures; ++i) {
        vecs3D[i] = cv::Vec3d(x(cv::Rect(0, 13 + 3 * i, 1, 3)));
    }

    CameraPose pose;

    for (const cv::Vec3d& v : vecs3D) {
        const std::optional<cv::Point2d> pOpt = cameraModel_.project(pose, v);

        if (pOpt.has_value()) {
            const cv::Point2d p = pOpt.value();

            if (cameraModel_.inView(p)) {
            }
        }
    }
}

cv::Mat1d
monoslam::SystemObservationModel::computeH(const cv::Mat1d& x) const
{
    // Compute the camera rotation (from world basis to camera basis)
    const cv::Matx33d R = quaternion::toMatrix(x(cv::Rect(0, 3, 1, 4))).t();

    // Get the camera position (in world coordinates)
    const cv::Mat1d t = x(cv::Rect(0, 0, 1, 3));

    const int numFeatures = (x.rows - 13) / 3;

    std::vector<cv::Point3d> points3D(static_cast<size_t>(numFeatures));

    // Create a vector of feature positions
    for (size_t i = 0; i < points3D.size(); ++i) {
        points3D[i] = cv::Point3d(x(cv::Rect(0, 13 + 3 * i, 1, 3)));
    }

    std::vector<cv::Point2d> points2D;

    // Project all the feature positions onto the current view
    camera.projectPoints(R, t, points3D_, points2D_);

    vector<Point3d> points3D;

    Rect box(Point2i(0, 0), camera.frameSize);

    for (unsigned int i = 0; i < points2D_.size(); i++) {
        if (box.contains(points2D_[i])) {
            inviewIndices.push_back(i);
            inviewPos.push_back(points2D_[i]);

            points3D.push_back(points3D_[i]);
        }
    }

    Mat H(2 * inviewIndices.size(), x.rows, CV_64FC1, Scalar(0));

    Mat R1, R2, R3, R4;

    toMatrixDerivatives(x(Rect(0, 3, 1, 4)), R1, R2, R3, R4);

    double fx = camera.K.at<double>(0, 0);
    double fy = camera.K.at<double>(1, 1);

    double k1 = camera.distCoeffs.at<double>(0, 0);
    double k2 = camera.distCoeffs.at<double>(0, 1);
    double p1 = camera.distCoeffs.at<double>(0, 2);
    double p2 = camera.distCoeffs.at<double>(0, 3);
    double k3 = camera.distCoeffs.at<double>(0, 4);

    for (int i = 0; i < H.rows / 2; i++) {
        Mat pt = Mat(points3D[i]) - t;

        Mat pCam = R * pt;

        double xCam = pCam.at<double>(0, 0);
        double yCam = pCam.at<double>(1, 0);
        double zCam = pCam.at<double>(2, 0);

        double zCam2 = zCam * zCam;

        Mat pBar1 = R1 * pt;
        Mat pBar2 = R2 * pt;
        Mat pBar3 = R3 * pt;
        Mat pBar4 = R4 * pt;

        // D(xn)/D(r): derivatives of normalized camera coordinate xn with respect to camera position
        double dxndr1 = (R.at<double>(2, 0) * xCam - R.at<double>(0, 0) * zCam) / zCam2;
        double dxndr2 = (R.at<double>(2, 1) * xCam - R.at<double>(0, 1) * zCam) / zCam2;
        double dxndr3 = (R.at<double>(2, 2) * xCam - R.at<double>(0, 2) * zCam) / zCam2;

        // D(xn)/D(q): derivatives of normalized camera coordinate xn with respect to camera orientation
        double dxndq1 = (pBar1.at<double>(0, 0) * zCam - pBar1.at<double>(2, 0) * xCam) / zCam2;
        double dxndq2 = (pBar2.at<double>(0, 0) * zCam - pBar2.at<double>(2, 0) * xCam) / zCam2;
        double dxndq3 = (pBar3.at<double>(0, 0) * zCam - pBar3.at<double>(2, 0) * xCam) / zCam2;
        double dxndq4 = (pBar4.at<double>(0, 0) * zCam - pBar4.at<double>(2, 0) * xCam) / zCam2;

        // D(yn)/D(r): derivatives of normalized camera coordinate yn with respect to camera position
        double dyndr1 = (R.at<double>(2, 0) * yCam - R.at<double>(1, 0) * zCam) / zCam2;
        double dyndr2 = (R.at<double>(2, 1) * yCam - R.at<double>(1, 1) * zCam) / zCam2;
        double dyndr3 = (R.at<double>(2, 2) * yCam - R.at<double>(1, 2) * zCam) / zCam2;

        // D(yn)/D(q): derivatives of normalized camera coordinate yn with respect to camera orientation
        double dyndq1 = (pBar1.at<double>(1, 0) * zCam - pBar1.at<double>(2, 0) * yCam) / zCam2;
        double dyndq2 = (pBar2.at<double>(1, 0) * zCam - pBar2.at<double>(2, 0) * yCam) / zCam2;
        double dyndq3 = (pBar3.at<double>(1, 0) * zCam - pBar3.at<double>(2, 0) * yCam) / zCam2;
        double dyndq4 = (pBar4.at<double>(1, 0) * zCam - pBar4.at<double>(2, 0) * yCam) / zCam2;

        double xn = xCam / zCam;
        double yn = yCam / zCam;

        // D(r2)/D(r): derivatives of r^2 = xn^2 + yn^2 with respect to camera position
        double dr2dr1 = 2 * (xn * dxndr1 + yn * dyndr1);
        double dr2dr2 = 2 * (xn * dxndr2 + yn * dyndr2);
        double dr2dr3 = 2 * (xn * dxndr3 + yn * dyndr3);

        // D(r2)/D(q): derivatives of r^2 = xn^2 + yn^2 with respect to camera orientation
        double dr2dq1 = 2 * (xn * dxndq1 + yn * dyndq1);
        double dr2dq2 = 2 * (xn * dxndq2 + yn * dyndq2);
        double dr2dq3 = 2 * (xn * dxndq3 + yn * dyndq3);
        double dr2dq4 = 2 * (xn * dxndq4 + yn * dyndq4);

        double r2 = xn * xn + yn * yn;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double a = 1 + k1 * r2 + k2 * r4 + k3 * r6;
        double b = k1 + 2 * k2 * r2 + 3 * k3 * r4;

        double ddr1 = 2 * (xn * dyndr1 + yn * dxndr1);
        double ddr2 = 2 * (xn * dyndr2 + yn * dxndr2);
        double ddr3 = 2 * (xn * dyndr3 + yn * dxndr3);

        double ddq1 = 2 * (xn * dyndq1 + yn * dxndq1);
        double ddq2 = 2 * (xn * dyndq2 + yn * dxndq2);
        double ddq3 = 2 * (xn * dyndq3 + yn * dxndq3);
        double ddq4 = 2 * (xn * dyndq4 + yn * dxndq4);

        // D(u)/D(r)
        double dudr1 = fx * (dxndr1 * a + xn * dr2dr1 * b + p1 * ddr1 + p2 * (dr2dr1 + 4 * xn * dxndr1));
        double dudr2 = fx * (dxndr2 * a + xn * dr2dr2 * b + p1 * ddr2 + p2 * (dr2dr2 + 4 * xn * dxndr2));
        double dudr3 = fx * (dxndr3 * a + xn * dr2dr3 * b + p1 * ddr3 + p2 * (dr2dr3 + 4 * xn * dxndr3));

        // D(u)/D(x)
        double dudx1 = -dudr1;
        double dudx2 = -dudr2;
        double dudx3 = -dudr3;

        // D(u)/D(q)
        double dudq1 = fx * (dxndq1 * a + xn * dr2dq1 * b + p1 * ddq1 + p2 * (dr2dq1 + 4 * xn * dxndq1));
        double dudq2 = fx * (dxndq2 * a + xn * dr2dq2 * b + p1 * ddq2 + p2 * (dr2dq2 + 4 * xn * dxndq2));
        double dudq3 = fx * (dxndq3 * a + xn * dr2dq3 * b + p1 * ddq3 + p2 * (dr2dq3 + 4 * xn * dxndq3));
        double dudq4 = fx * (dxndq4 * a + xn * dr2dq4 * b + p1 * ddq4 + p2 * (dr2dq4 + 4 * xn * dxndq4));

        // D(v)/D(r)
        double dvdr1 = fy * (dyndr1 * a + yn * dr2dr1 * b + p2 * ddr1 + p1 * (dr2dr1 + 4 * yn * dyndr1));
        double dvdr2 = fy * (dyndr2 * a + yn * dr2dr2 * b + p2 * ddr2 + p1 * (dr2dr2 + 4 * yn * dyndr2));
        double dvdr3 = fy * (dyndr3 * a + yn * dr2dr3 * b + p2 * ddr3 + p1 * (dr2dr3 + 4 * yn * dyndr3));

        // D(v)/D(x)
        double dvdx1 = -dvdr1;
        double dvdx2 = -dvdr2;
        double dvdx3 = -dvdr3;

        // D(v)/D(q)
        double dvdq1 = fy * (dyndq1 * a + yn * dr2dq1 * b + p2 * ddq1 + p1 * (dr2dq1 + 4 * yn * dyndq1));
        double dvdq2 = fy * (dyndq2 * a + yn * dr2dq2 * b + p2 * ddq2 + p1 * (dr2dq2 + 4 * yn * dyndq2));
        double dvdq3 = fy * (dyndq3 * a + yn * dr2dq3 * b + p2 * ddq3 + p1 * (dr2dq3 + 4 * yn * dyndq3));
        double dvdq4 = fy * (dyndq4 * a + yn * dr2dq4 * b + p2 * ddq4 + p1 * (dr2dq4 + 4 * yn * dyndq4));

        H.at<double>(2 * i, 0) = dudr1;
        H.at<double>(2 * i, 1) = dudr2;
        H.at<double>(2 * i, 2) = dudr3;
        H.at<double>(2 * i, 3) = dudq1;
        H.at<double>(2 * i, 4) = dudq2;
        H.at<double>(2 * i, 5) = dudq3;
        H.at<double>(2 * i, 6) = dudq4;

        H.at<double>(2 * i, 13 + 3 * i) = dudx1;
        H.at<double>(2 * i, 13 + 3 * i + 1) = dudx2;
        H.at<double>(2 * i, 13 + 3 * i + 2) = dudx3;

        H.at<double>(2 * i + 1, 0) = dvdr1;
        H.at<double>(2 * i + 1, 1) = dvdr2;
        H.at<double>(2 * i + 1, 2) = dvdr3;
        H.at<double>(2 * i + 1, 3) = dvdq1;
        H.at<double>(2 * i + 1, 4) = dvdq2;
        H.at<double>(2 * i + 1, 5) = dvdq3;
        H.at<double>(2 * i + 1, 6) = dvdq4;

        H.at<double>(2 * i + 1, 13 + 3 * i) = dvdx1;
        H.at<double>(2 * i + 1, 13 + 3 * i + 1) = dvdx2;
        H.at<double>(2 * i + 1, 13 + 3 * i + 2) = dvdx3;
    }

    return H;
}
