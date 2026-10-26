#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "camera.h"
#include "feature.h"
#include "filter/ekf.h"
#include "filter/observation_model.h"
#include "matcher.h"
#include "system_observation_model.h"
#include "system_process_model.h"

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

    const cv::Size patternSize(cols - 1, rows - 1);

    cv::Mat1d x;
    cv::Mat1d P;

    std::vector<monoslam::Feature> features;

    Clock::time_point t0;

    cv::Mat3b frame;

    do {
        if (!device.read(frame)) {
            std::cerr << "Could not read a camera frame\n";
            continue;
        }

        t0 = Clock::now();

        const bool ready = monoslam::initSystem(frame, patternSize, x, P, features);

    } while (!ready);

    constexpr double linearAccStd = 50.0;   // Linear acceleration standard deviation, in cm / s²
    constexpr double angularAccStd = 1.57;  // Angular acceleration standard deviation, in rad / s²

    const auto processModel = std::make_shared<monoslam::SystemProcessModel>(linearAccStd, angularAccStd);

    const monoslam::CameraModel camera(K, distCoeffs, frameSize);

    constexpr double pixelStd = 2.0;

    const auto observationModel = std::make_shared<monoslam::SystemObservationModel>(camera, pixelStd);

    monoslam::filter::EKF ekf(processModel, observationModel);
    ekf.init(x, P);

    monoslam::Matcher matcher;

    cv::Mat1b gray;

    while (true) {
        if (!device.read(frame)) {
            std::cerr << "Could not read a camera frame\n";
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        Clock::time_point t1 = Clock::now();

        double dt = std::chrono::duration<double>(t1 - t0).count();

        ekf.predict(dt);

        ekf.update([&](const cv::Mat1d& x, const monoslam::filter::ObservationPrediction& p, const cv::Mat1d& S) {
            return matcher.match(gray, features, x, p, S);
        });

        t0 = t1;
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
