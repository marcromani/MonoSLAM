#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/calib.hpp>  // TODO: Better to import calib3d?
#include <opencv2/objdetect.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

bool
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, double& tol, std::string& filename);

void
printUsage(const char* execName);

std::vector<cv::Point3d>
build3DCoordinates(const cv::Size& patternSize, double squareSize);

int
main(int argc, char* argv[])
{
    int rows, cols;
    double squareSize;
    std::string filename;

    double tol = 1.5;  // Tolerance for the sample reprojection error, in pixels

    if (!getOptions(argc, argv, rows, cols, squareSize, tol, filename)) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "Calibrating with a " << cols << "x" << rows << " chessboard with a square size of " << 10 * squareSize
              << " cm" << std::endl
              << "Sample reprojection error tolerance is set to " << tol << " pixels" << std::endl;

    const cv::Size patternSize(cols - 1, rows - 1);

    // Initialize video feed device
    cv::VideoCapture device("/dev/video2");

    if (!device.isOpened()) {
        return EXIT_FAILURE;
    }

    device.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    device.set(cv::CAP_PROP_FRAME_HEIGHT, 768);

    const cv::Size frameSize(640, 480);

    // Create a window
    cv::namedWindow("Camera calibration", cv::WINDOW_AUTOSIZE);

    std::vector<std::vector<cv::Point2d>> lists2D;

    int i = 0;
    std::chrono::time_point<Clock> t0 = Clock::now();

    cv::Mat3b showFrame(480, 2 * 640, cv::Vec3b::zeros());

    while (true) {
        cv::Mat3b frame;
        device.read(frame);
        cv::resize(frame, frame, frameSize);

        frame.copyTo(showFrame(cv::Rect(0, 0, frame.cols, frame.rows)));

        const double elapsed = (Clock::now() - t0).count();

        // Leave some seconds between each pattern capture
        if (elapsed > 2) {
            std::vector<cv::Point2d> points2D;

            const bool patternFound =
              cv::findChessboardCorners(frame,
                                        patternSize,
                                        points2D,
                                        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
                                          cv::CALIB_CB_FILTER_QUADS + cv::CALIB_CB_FAST_CHECK);

            if (patternFound) {
                cv::Mat1b gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

                cv::cornerSubPix(gray, points2D, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(30, 0.1));

                lists2D.push_back(points2D);

                cv::drawChessboardCorners(
                  showFrame(cv::Rect(frame.cols, 0, frame.cols, frame.rows)), patternSize, points2D, true);

                std::cout << "\r" << ++i << " samples captured" << std::flush;
                t0 = Clock::now();

            } else {
                showFrame(cv::Rect(frame.cols, 0, frame.cols, frame.rows)) = cv::Vec3b::zeros();
            }
        }

        cv::imshow("Camera calibration", showFrame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    if (i == 0) {
        std::cout << "Calibration canceled: no samples to work with" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << std::endl;

    const std::vector<cv::Point3d> points3D = build3DCoordinates(patternSize, squareSize);

    std::vector<std::vector<cv::Point3d>> lists3D;
    lists3D.reserve(lists2D.size());

    for (const std::vector<cv::Point2d>& points2D : lists2D) {
        lists3D.push_back(points3D);
    }

    cv::Matx33d K;
    cv::Vec<double, 5> distCoeffs;
    std::vector<cv::Vec3d> rotationVecs;
    std::vector<cv::Vec3d> translationVecs;
    std::vector<double> stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors;

    double rmse = cv::calibrateCamera(lists3D,
                                      lists2D,
                                      frameSize,
                                      K,
                                      distCoeffs,
                                      rotationVecs,
                                      translationVecs,
                                      stdDeviationIntrinsics,
                                      stdDeviationExtrinsics,
                                      perViewErrors);

    // Remove samples with high reprojection error
    std::size_t idx = 0;

    for (std::size_t i = 0; i < perViewErrors.size(); ++i) {
        if (perViewErrors[i] < tol) {
            std::swap(lists2D[idx], lists2D[i]);
            std::swap(lists3D[idx], lists3D[i]);
            std::swap(perViewErrors[idx], perViewErrors[i]);

            ++idx;
        }
    }

    lists2D.resize(idx);
    lists3D.resize(idx);

    if (lists2D.empty()) {
        std::cout << "Calibration canceled: no good samples to work with" << std::endl;
        return EXIT_FAILURE;
    }

    if (lists2D.size() < perViewErrors.size()) {
        // Recalibrate with good samples
        rmse = cv::calibrateCamera(lists3D, lists2D, frameSize, K, distCoeffs, rotationVecs, translationVecs);

        std::cout << "Camera was calibrated using " << lists2D.size() << " samples" << std::endl;
    }

    std::cout << "Intrinsic matrix:" << std::endl
              << K << std::endl
              << "Distortion coeficients: " << distCoeffs << std::endl
              << "RMS reprojection error (pixels): " << rmse << std::endl;

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cout << "Could not open \"" << filename << "\": camera settings not saved" << std::endl;
        return EXIT_FAILURE;
    }

    fs << "Image_Size" << frameSize;
    fs << "Intrinsic_Matrix" << K;
    fs << "Distortion_Coefficients" << distCoeffs;

    std::cout << "Camera settings saved to \"" << filename << "\"" << std::endl;

    return EXIT_SUCCESS;
}

bool
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, double& tol, std::string& filename)
{
    // Look for the help option
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
            return false;
        }
    }

    // Check the number of input options
    if (argc != 5 && argc != 6) {
        return false;
    }

    rows = static_cast<int>(strtol(argv[1], NULL, 10));
    if (rows == 0 || errno == ERANGE) {
        return false;
    }

    cols = static_cast<int>(strtol(argv[2], NULL, 10));
    if (cols == 0 || errno == ERANGE) {
        return false;
    }

    squareSize = 0.1 * strtod(argv[3], NULL);
    if (squareSize == 0 || errno == ERANGE) {
        return false;
    }

    if (argc == 6) {
        tol = strtod(argv[4], NULL);
        if (tol == 0 || errno == ERANGE) {
            return false;
        }
    }

    filename = (argc == 6) ? argv[5] : argv[4];
    filename += std::string(".json");

    return true;
}

void
printUsage(const char* execName)
{
    std::cerr << "Usage: " << execName << " rows cols square_size (cm) [tol] filename (-h, --help for help)"
              << std::endl;
}

std::vector<cv::Point3d>
build3DCoordinates(const cv::Size& patternSize, double squareSize)
{
    std::vector<cv::Point3d> points3D;
    points3D.reserve(patternSize.width * patternSize.height);

    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            points3D.emplace_back(squareSize * i, squareSize * j, 0.0);
        }
    }

    return points3D;
}
