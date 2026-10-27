#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include "camera.h"
#include "feature.h"
#include "filter/ekf.h"
#include "filter/observation_model.h"
#include "matcher.h"
#include "quaternion.h"
#include "system_observation_model.h"
#include "system_process_model.h"
#include "util.h"

using Clock = std::chrono::steady_clock;

bool
parseInt(std::string_view text, int& value);

bool
parsePositiveDouble(std::string_view text, double& value);

bool
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, std::string& filename);

void
printUsage(const char* execName);

bool
readCameraFile(const std::string& filename, cv::Matx33d& K, cv::Vec<double, 5>& distCoeffs, cv::Size& frameSize);

bool
initSystem(const cv::Mat1b& frame,
           const cv::Size& patternSize,
           double squareSize,
           const monoslam::CameraModel& camera,
           cv::Mat1d& x,
           cv::Mat1d& P,
           std::vector<monoslam::Feature>& features);

bool
addInitialFeature(const cv::Mat1b& frame,
                  const cv::Point2f& point2D,
                  const cv::Matx33d& R,
                  const cv::Vec3d& t,
                  std::vector<monoslam::Feature>& features);

int
main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    int rows = 0, cols = 0;
    double squareSize = 0.0;
    std::string filename;

    if (!getOptions(argc, argv, rows, cols, squareSize, filename)) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const cv::Size patternSize(cols - 1, rows - 1);

    cv::Matx33d K;
    cv::Vec<double, 5> distCoeffs;
    cv::Size frameSize;

    if (!readCameraFile(filename, K, distCoeffs, frameSize)) {
        std::cerr << "Could not open \"" << filename << "\": camera settings not loaded\n";
        return EXIT_FAILURE;
    }

    cv::VideoCapture device("/dev/video4");

    if (!device.isOpened()) {
        std::cerr << "Could not open /dev/video4\n";
        return EXIT_FAILURE;
    }

    cv::namedWindow("MonoSLAM", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);

    cv::Mat1d x;
    cv::Mat1d P;

    std::vector<monoslam::Feature> features;

    Clock::time_point t0;

    const monoslam::CameraModel camera(K, distCoeffs, frameSize);

    cv::Mat3b frame;
    cv::Mat1b gray;

    do {
        if (!device.read(frame)) {
            std::cerr << "Could not read a camera frame\n";
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        t0 = Clock::now();

        cv::imshow("MonoSLAM", gray);

        if (cv::waitKey(1) == 27) {
            break;
        }

    } while (!initSystem(gray, patternSize, squareSize, camera, x, P, features));

    std::cout << "Initialization succeeded\n";

    constexpr double linearAccStd = 50.0;   // Linear acceleration standard deviation, in cm / s²
    constexpr double angularAccStd = 1.57;  // Angular acceleration standard deviation, in rad / s²

    const auto processModel = std::make_shared<monoslam::SystemProcessModel>(linearAccStd, angularAccStd);

    constexpr double pixelStd = 2.0;

    const auto observationModel = std::make_shared<monoslam::SystemObservationModel>(camera, pixelStd);

    monoslam::filter::EKF ekf(processModel, observationModel);
    ekf.init(x, P);

    monoslam::Matcher matcher;

    while (true) {
        if (!device.read(frame)) {
            std::cerr << "Could not read a camera frame\n";
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        Clock::time_point t1 = Clock::now();

        double dt = std::chrono::duration<double>(t1 - t0).count();

        ekf.predict(dt);

        ekf.update([&](const cv::Mat1d& x_, const monoslam::filter::ObservationPrediction& p_, const cv::Mat1d& S_) {
            const monoslam::filter::Measurement measurement = matcher.match(gray, features, K, x_, p_, S_);

            for (int i = 0; i < measurement.z.rows; i += 2) {
                const double u = measurement.z(i);
                const double v = measurement.z(i + 1);

                const cv::Point center(static_cast<int>(u), static_cast<int>(v));

                monoslam::util::drawCircle(gray, center, 10, cv::Scalar(0));
            }

            std::cout << measurement.z.rows << std::endl;

            return measurement;
        });

        t0 = t1;

        cv::imshow("MonoSLAM", gray);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return EXIT_SUCCESS;
}

bool
parseInt(std::string_view text, int& value)
{
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc() && end == text.data() + text.size();
}

bool
parsePositiveDouble(std::string_view text, double& value)
{
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc() && end == text.data() + text.size() && std::isfinite(value) && value > 0.0;
}

bool
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, std::string& filename)
{
    if (argc != 5) {
        return false;
    }

    if (!parseInt(argv[1], rows) || !parseInt(argv[2], cols)) {
        return false;
    }

    if (!parsePositiveDouble(argv[3], squareSize)) {
        return false;
    }

    filename = argv[argc - 1];
    filename += ".json";

    return true;
}

void
printUsage(const char* execName)
{
    std::cerr << "Usage: " << execName << " rows cols square_size filename\n"
              << "  rows, cols: Number of squares of the calibration pattern\n"
              << "  square_size: Size of the square, in cm\n"
              << "  filename: Camera intrinsics filename, \".json\" is appended\n"
              << "  -h, --help: Show this message\n";
}

bool
readCameraFile(const std::string& filename, cv::Matx33d& K, cv::Vec<double, 5>& distCoeffs, cv::Size& frameSize)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        return false;
    }

    cv::FileNode node = fs["fx"];
    CV_Assert(!node.empty());

    double fx;
    node >> fx;

    node = fs["fy"];
    CV_Assert(!node.empty());

    double fy;
    node >> fy;

    node = fs["cx"];
    CV_Assert(!node.empty());

    double cx;
    node >> cx;

    node = fs["cy"];
    CV_Assert(!node.empty());

    double cy;
    node >> cy;

    node = fs["k1"];
    CV_Assert(!node.empty());

    double k1;
    node >> k1;

    node = fs["k2"];
    CV_Assert(!node.empty());

    double k2;
    node >> k2;

    node = fs["p1"];
    CV_Assert(!node.empty());

    double p1;
    node >> p1;

    node = fs["p2"];
    CV_Assert(!node.empty());

    double p2;
    node >> p2;

    node = fs["k3"];
    CV_Assert(!node.empty());

    double k3;
    node >> k3;

    node = fs["Image size"];
    CV_Assert(!node.empty());

    node >> frameSize;

    fs.release();

    K = cv::Matx33d(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    distCoeffs = cv::Vec<double, 5>(k1, k2, p1, p2, k3);

    return true;
}

bool
initSystem(const cv::Mat1b& frame,
           const cv::Size& patternSize,
           double squareSize,
           const monoslam::CameraModel& camera,
           cv::Mat1d& x,
           cv::Mat1d& P,
           std::vector<monoslam::Feature>& features)
{
    std::vector<cv::Point2f> points2D;

    const bool patternFound = cv::findChessboardCorners(frame,
                                                        patternSize,
                                                        points2D,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                                          cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK);

    if (patternFound) {
        cv::cornerSubPix(frame, points2D, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(30, 0.1));

    } else {
        return false;
    }

    std::vector<cv::Point3f> points3D;
    points3D.reserve(static_cast<std::size_t>(patternSize.width * patternSize.height));

    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            points3D.emplace_back(static_cast<float>(squareSize * i), static_cast<float>(squareSize * j), 0.0f);
        }
    }

    cv::Vec3d rotationVec, translationVec;

    const bool poseFound = cv::solvePnP(
      points3D, points2D, camera.K(), camera.distCoeffs(), rotationVec, translationVec, false, cv::SOLVEPNP_ITERATIVE);

    if (!poseFound) {
        return false;
    }

    cv::Matx33d R;
    cv::Vec3d t;

    // Convert to rotation R (from world to camera) and translation t (in world coordinates)
    cv::Rodrigues(rotationVec, R);
    t = -R.t() * translationVec;

    // Compute the camera orientation quaternion
    const cv::Vec4d q = monoslam::quaternion::conjugate(monoslam::quaternion::fromVector(rotationVec));

    // ADD CAMERA //

    // Update the camera state (zero linear and angular velocities)
    x = cv::Mat1d::zeros(13 + 3 * 5, 1);

    t.copyTo(x(cv::Rect(0, 0, 1, 3)));
    q.copyTo(x(cv::Rect(0, 3, 1, 4)));

    // Update the camera covariance
    P = cv::Mat1d::zeros(13 + 3 * 5, 13 + 3 * 5);

    P(0, 0) = 1.0;
    P(1, 1) = 1.0;
    P(2, 2) = 1.0;

    P(3, 3) = 0.0;
    P(4, 4) = 0.0;
    P(5, 5) = 0.0;
    P(6, 6) = 0.0;

    P(7, 7) = 1.0;
    P(8, 8) = 1.0;
    P(9, 9) = 1.0;

    P(10, 10) = 0.00762;
    P(11, 11) = 0.00762;
    P(12, 12) = 0.00762;

    // ADD INITIAL FEATURES //

    features.clear();

    const size_t w = static_cast<size_t>(patternSize.width);
    const size_t h = static_cast<size_t>(patternSize.height);

    size_t idx = 0;

    if (!addInitialFeature(frame, points2D[idx], R, t, features)) {
        return false;
    }

    x(13) = points3D[idx].x;
    x(14) = points3D[idx].y;
    x(15) = points3D[idx].z;

    P(13, 13) = 1.0;
    P(14, 14) = 1.0;
    P(15, 15) = 1.0;

    idx = w - 1;

    if (!addInitialFeature(frame, points2D[idx], R, t, features)) {
        return false;
    }

    x(16) = points3D[idx].x;
    x(17) = points3D[idx].y;
    x(18) = points3D[idx].z;

    P(16, 16) = 1.0;
    P(17, 17) = 1.0;
    P(18, 18) = 1.0;

    idx = (h - 1) * w;

    if (!addInitialFeature(frame, points2D[idx], R, t, features)) {
        return false;
    }

    x(19) = points3D[idx].x;
    x(20) = points3D[idx].y;
    x(21) = points3D[idx].z;

    P(19, 19) = 1.0;
    P(20, 20) = 1.0;
    P(21, 21) = 1.0;

    idx = w * h - 1;

    if (!addInitialFeature(frame, points2D[idx], R, t, features)) {
        return false;
    }

    x(22) = points3D[idx].x;
    x(23) = points3D[idx].y;
    x(24) = points3D[idx].z;

    P(22, 22) = 1.0;
    P(23, 23) = 1.0;
    P(24, 24) = 1.0;

    idx = w * ((h - 1) / 2) + (w + 1) / 2 - 1;

    if (!addInitialFeature(frame, points2D[idx], R, t, features)) {
        return false;
    }

    x(25) = points3D[idx].x;
    x(26) = points3D[idx].y;
    x(27) = points3D[idx].z;

    P(25, 25) = 1.0;
    P(26, 26) = 1.0;
    P(27, 27) = 1.0;

    return true;
}

bool
addInitialFeature(const cv::Mat1b& frame,
                  const cv::Point2f& point2D,
                  const cv::Matx33d& R,
                  const cv::Vec3d& t,
                  std::vector<monoslam::Feature>& features)
{
    const cv::Point pixel(static_cast<int>(point2D.x), static_cast<int>(point2D.y));
    const cv::Rect roi = monoslam::util::buildSquare(pixel, 23);

    const bool overflow = roi.x < 0 || roi.x + roi.width > frame.cols || roi.y < 0 || roi.y + roi.height > frame.rows;

    // Check if the feature patch is outside the frame boundaries
    if (overflow) {
        return false;
    }

    const cv::Vec3d normal(0.0, 0.0, 1.0);

    // Add the feature
    features.emplace_back(frame, roi, normal, R, t);

    return true;
}
