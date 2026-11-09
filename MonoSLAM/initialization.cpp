#include "initialization.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "ekf.h"
#include "inverse_depth.h"

namespace {

constexpr double maxReprojectionRms = 1.0;    // Pixels
constexpr double maxReprojectionError = 3.0;  // Pixels

// Initial uncertainty
constexpr double positionVariance = 0.05;     // cm^2
constexpr double orientationVariance = 4e-4;  // rad^2
constexpr double linearVelStd = 30.0;         // cm/s, the camera may already be moving
constexpr double angularVelStd = 1.5;         // rad/s
constexpr double featureVariance = 0.05;      // cm^2

/*
 * Returns the corners of the pattern that become features: the four outermost ones, and the one closest to the center.
 *
 * patternSize    Number of inner corners of the pattern, per row and per column
 */
std::vector<std::size_t>
selectCorners(const cv::Size& patternSize)
{
    const auto w = static_cast<std::size_t>(patternSize.width);
    const auto h = static_cast<std::size_t>(patternSize.height);

    return { 0, w - 1, (h - 1) * w, w * h - 1, w * ((h - 1) / 2) + (w + 1) / 2 - 1 };
}

}

bool
monoslam::initializeFromPattern(const cv::Mat1b& frame,
                                const cv::Size& patternSize,
                                double squareSize,
                                const CameraModel& camera,
                                cv::Mat1d& x,
                                cv::Matx33d& orientation,
                                cv::Mat1d& P,
                                std::vector<Feature>& features)
{
    std::vector<cv::Point2f> points2D;

    const bool patternFound = cv::findChessboardCorners(frame,
                                                        patternSize,
                                                        points2D,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                                          cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK);

    if (!patternFound) {
        return false;
    }

    cv::cornerSubPix(frame, points2D, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(100, 0.1));

    std::vector<cv::Point3f> points3D;
    points3D.reserve(static_cast<std::size_t>(patternSize.width * patternSize.height));

    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            points3D.emplace_back(static_cast<float>(squareSize * i), static_cast<float>(squareSize * j), 0.0f);
        }
    }

    // CAMERA POSE //

    cv::Vec3d rotationVec, translationVec;

    const bool poseFound = cv::solvePnP(
      points3D, points2D, camera.K(), camera.distCoeffs(), rotationVec, translationVec, false, cv::SOLVEPNP_ITERATIVE);

    if (!poseFound) {
        return false;
    }

    cv::solvePnPRefineLM(
      points3D, points2D, camera.K(), camera.distCoeffs(), rotationVec, translationVec, cv::TermCriteria(100, 1e-8));

    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(points3D, rotationVec, translationVec, camera.K(), camera.distCoeffs(), reprojected);

    double reprojectionRMS = 0.0;
    double reprojectionMax = 0.0;

    for (std::size_t i = 0; i < points2D.size(); ++i) {
        const double error = cv::norm(points2D[i] - reprojected[i]);
        reprojectionRMS += error * error;
        reprojectionMax = std::max(reprojectionMax, error);
    }

    reprojectionRMS = std::sqrt(reprojectionRMS / static_cast<double>(points2D.size()));

    std::cout << "Initialization reprojection error: " << reprojectionRMS << " pixels (RMS), " << reprojectionMax
              << " pixels (Max)\n";

    if (!std::isfinite(reprojectionRMS) ||
        !std::isfinite(reprojectionMax) ||
        reprojectionRMS > maxReprojectionRms ||
        reprojectionMax > maxReprojectionError) {
        return false;
    }

    // Camera orientation and position: solvePnP gives the rotation from world to camera
    orientation = rotation(rotationVec).t();

    const CameraPose pose(-orientation * translationVec, orientation);
    const cv::Vec3d& t = pose.r();

    // STATE //

    const std::vector<std::size_t> corners = selectCorners(patternSize);

    x = cv::Mat1d::zeros(cameraStateSize + featureSize * static_cast<int>(corners.size()), 1);
    P = cv::Mat1d::zeros(x.rows, x.rows);

    t.copyTo(x(cv::Rect(0, 0, 1, 3)));

    for (int i = 0; i < 3; ++i) {
        P(i, i) = positionVariance;
    }

    for (int i = 3; i < 6; ++i) {
        P(i, i) = orientationVariance;
    }

    for (int i = 6; i < 9; ++i) {
        P(i, i) = linearVelStd * linearVelStd;
    }

    for (int i = 9; i < cameraStateSize; ++i) {
        P(i, i) = angularVelStd * angularVelStd;
    }

    // FEATURES //

    features.clear();

    const cv::Vec3d normal(0.0, 0.0, 1.0);  // The normal of the pattern, in world coordinates

    for (std::size_t i = 0; i < corners.size(); ++i) {
        const cv::Point2f& pixel = points2D[corners[i]];
        const cv::Point3f& point = points3D[corners[i]];

        Feature feature(frame, pixel, normal, pose, i);
        feature.knownSurface = true;

        if ((feature.roi & cv::Rect(0, 0, frame.cols, frame.rows)) != feature.roi) {
            return false;
        }

        features.push_back(feature);

        // The corner, seen from the initial camera position, which stays fixed as its anchor. The uncertainty
        // of its position is carried by the angles and the inverse depth
        const cv::Vec3d corner(point.x, point.y, point.z);
        const inverse_depth::Parameters y = inverse_depth::fromPoint(t, corner);
        const cv::Matx33d J = inverse_depth::fromPointJacobian(t, corner);

        const int idx = featureIndex(i);

        for (int k = 0; k < featureSize; ++k) {
            x(idx + k) = y[k];
        }

        cv::Mat1d(featureVariance * J * J.t()).copyTo(P(cv::Rect(idx + 3, idx + 3, 3, 3)));
    }

    return true;
}
