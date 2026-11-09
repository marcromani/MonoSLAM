/*
 * analyze_pattern: the ground truth of a recorded sequence, from the calibration pattern, for the evaluation
 * of monoslam only. It finds the whole pattern, 9 x 6 inner corners, in every frame, fits the camera pose to its 54
 * corners, and prints for every frame:
 *
 *   - rms=, the RMS reprojection error of the corners, in pixels. Near the border of the frame, where the lens model
 *     is least accurate, it can reach several pixels, and the pose is then less reliable
 *   - pos=, the camera position, in world coordinates
 *   - rot=, the camera orientation, as the rotation vector from camera to world
 *   - sel=, the pixels of the five corners that monoslam initializes as features
 *
 * Usage: analyze_pattern sequence > ground_truth.txt
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "camera.h"

namespace {

const cv::Size patternSize(9, 6);   // Inner corners
constexpr double squareSize = 2.5;  // cm

// The corners that monoslam initializes as features, see initialization.cpp
const std::vector<std::size_t> selected = { 0, 8, 45, 53, 22 };

}

int
main(int argc, char* argv[])
{
    namespace fs = std::filesystem;

    if (argc != 2) {
        std::cerr << "Usage: analyze_pattern sequence > ground_truth.txt\n";
        return EXIT_FAILURE;
    }

    const fs::path sequence(argv[1]);
    const std::optional<monoslam::CameraModel> camera =
      monoslam::CameraModel::load((sequence / "camera.json").string());

    if (!camera.has_value()) {
        std::cerr << "Could not read " << (sequence / "camera.json").string() << '\n';
        return EXIT_FAILURE;
    }

    std::vector<std::string> files;

    for (const fs::directory_entry& entry : fs::directory_iterator(sequence)) {
        if (entry.path().extension() == ".png") {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());

    // The corners in the plane of the pattern, which is the plane z = 0 of the world
    std::vector<cv::Point3f> pattern;

    for (int row = 0; row < patternSize.height; ++row) {
        for (int col = 0; col < patternSize.width; ++col) {
            pattern.emplace_back(static_cast<float>(squareSize * row), static_cast<float>(squareSize * col), 0.0f);
        }
    }

    std::size_t numFound = 0;

    for (const std::string& file : files) {
        const std::string name = fs::path(file).filename().string();
        const cv::Mat1b frame = cv::imread(file, cv::IMREAD_GRAYSCALE);

        if (frame.empty()) {
            continue;
        }

        std::vector<cv::Point2f> corners;

        if (!cv::findChessboardCorners(frame,
                                       patternSize,
                                       corners,
                                       cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                         cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK)) {
            std::cout << name << " NOT_FOUND\n";
            continue;
        }

        cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(100, 0.1));

        // The rotation and translation from world to camera
        cv::Vec3d rotation, translation;

        if (!cv::solvePnP(pattern, corners, camera->K(), camera->distCoeffs(), rotation, translation)) {
            std::cout << name << " PNP_FAIL\n";
            continue;
        }

        cv::solvePnPRefineLM(pattern,
                             corners,
                             camera->K(),
                             camera->distCoeffs(),
                             rotation,
                             translation,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-8));

        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(pattern, rotation, translation, camera->K(), camera->distCoeffs(), reprojected);

        double rms = 0.0;

        for (std::size_t i = 0; i < corners.size(); ++i) {
            const double error = cv::norm(corners[i] - reprojected[i]);
            rms += error * error;
        }

        rms = std::sqrt(rms / static_cast<double>(corners.size()));

        // The camera pose, with its orientation from camera to world
        const cv::Matx33d R = monoslam::rotation(rotation).t();
        const cv::Vec3d r = -R * translation;

        cv::Vec3d orientation;
        cv::Rodrigues(R, orientation);

        std::cout << name << " OK rms=" << rms << " pos=[" << r[0] << "," << r[1] << "," << r[2] << "]";
        std::cout << " rot=[" << orientation[0] << "," << orientation[1] << "," << orientation[2] << "]\n  sel=[";

        for (std::size_t k = 0; k < selected.size(); ++k) {
            std::cout << (k == 0 ? "" : ",") << corners[selected[k]].x << "," << corners[selected[k]].y;
        }

        std::cout << "]\n";

        ++numFound;
    }

    std::cout << "FOUND=" << numFound << " TOTAL=" << files.size() << "\n";

    return EXIT_SUCCESS;
}
