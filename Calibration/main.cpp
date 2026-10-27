#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

using Clock = std::chrono::steady_clock;

bool
parseInt(std::string_view text, int& value);

bool
parsePositiveDouble(std::string_view text, double& value);

bool
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, double& tol, std::string& filename);

void
printUsage(const char* execName);

std::vector<cv::Point3f>
build3DCoordinates(const cv::Size& patternSize, double squareSize);

int
main(int argc, char* argv[])
try {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    int rows = 0, cols = 0;
    double squareSize = 0.0;
    double tol = 1.5;
    std::string filename;

    if (!getOptions(argc, argv, rows, cols, squareSize, tol, filename)) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "Calibrating with a " << cols << "x" << rows << " chessboard with a square size of " << squareSize
              << " cm\n"
              << "Per-view RMS rejection threshold set to " << tol << " pixels\n"
              << "Press Escape to finish capturing samples\n";

    const cv::Size patternSize(cols - 1, rows - 1);

    cv::VideoCapture device("/dev/video4");

    if (!device.isOpened()) {
        std::cerr << "Could not open /dev/video4\n";
        return EXIT_FAILURE;
    }

    const cv::Size frameSize(640, 480);

    cv::namedWindow("Camera calibration", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);

    cv::Mat3b showFrame(frameSize.height, 2 * frameSize.width, cv::Vec3b::zeros());
    cv::Mat3b left = showFrame(cv::Rect(0, 0, frameSize.width, frameSize.height));
    cv::Mat3b right = showFrame(cv::Rect(frameSize.width, 0, frameSize.width, frameSize.height));

    std::vector<std::vector<cv::Point2f>> lists2D;

    Clock::time_point t0 = Clock::now();

    while (true) {
        cv::Mat3b frame;
        if (!device.read(frame) || frame.empty()) {
            const char c = lists2D.empty() ? '\0' : '\n';
            std::cerr << c << "Could not read a camera frame\n";
            return EXIT_FAILURE;
        }

        cv::resize(frame, frame, frameSize);
        frame.copyTo(left);

        if (Clock::now() - t0 > std::chrono::seconds(2)) {
            std::vector<cv::Point2f> points2D;

            const bool patternFound =
              cv::findChessboardCorners(frame,
                                        patternSize,
                                        points2D,
                                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                          cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK);

            if (patternFound) {
                cv::Mat1b gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

                cv::cornerSubPix(gray, points2D, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(30, 0.1));

                lists2D.push_back(points2D);

                frame.copyTo(right);
                cv::drawChessboardCorners(right, patternSize, points2D, true);

                std::cout << '\r' << lists2D.size() << " samples captured" << std::flush;
                t0 = Clock::now();
            }
        }

        cv::imshow("Camera calibration", showFrame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    device.release();
    cv::destroyAllWindows();

    std::cout << '\n';

    if (lists2D.empty()) {
        std::cerr << "Calibration canceled: no samples to work with\n";
        return EXIT_FAILURE;
    }

    const std::vector<cv::Point3f> points3D = build3DCoordinates(patternSize, squareSize);
    std::vector<std::vector<cv::Point3f>> lists3D(lists2D.size(), points3D);

    cv::Matx33d K;
    cv::Vec<double, 5> distCoeffs;
    std::vector<cv::Vec3d> rotationVecs, translationVecs;
    std::vector<double> perViewErrors;

    double rmse = cv::calibrateCamera(lists3D,
                                      lists2D,
                                      frameSize,
                                      K,
                                      distCoeffs,
                                      rotationVecs,
                                      translationVecs,
                                      cv::noArray(),
                                      cv::noArray(),
                                      perViewErrors);

    // Reject once using the initial fit
    std::size_t kept = 0;
    for (std::size_t i = 0; i < perViewErrors.size(); ++i) {
        if (perViewErrors[i] < tol) {
            if (kept != i) {
                std::swap(lists2D[kept], lists2D[i]);
            }
            ++kept;
        }
    }

    lists2D.resize(kept);
    lists3D.resize(kept);

    if (lists2D.empty()) {
        std::cerr << "Calibration canceled: no good samples to work with\n";
        return EXIT_FAILURE;
    }

    if (kept < perViewErrors.size()) {
        rmse = cv::calibrateCamera(lists3D, lists2D, frameSize, K, distCoeffs, rotationVecs, translationVecs);
    }

    std::cout << "Camera was calibrated using " << kept << " of " << perViewErrors.size() << " samples\n"
              << "Pinhole parameters:\n"
              << "fx = " << K(0, 0) << '\n'
              << "fy = " << K(1, 1) << '\n'
              << "cx = " << K(0, 2) << '\n'
              << "cy = " << K(1, 2) << '\n'
              << "Lens parameters:\n"
              << "k1 = " << distCoeffs[0] << '\n'
              << "k2 = " << distCoeffs[1] << '\n'
              << "p1 = " << distCoeffs[2] << '\n'
              << "p2 = " << distCoeffs[3] << '\n'
              << "k3 = " << distCoeffs[4] << '\n'
              << "RMS reprojection error (pixels): " << rmse << '\n';

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cerr << "Could not open \"" << filename << "\": camera settings not saved\n";
        return EXIT_FAILURE;
    }

    fs << "fx" << K(0, 0);
    fs << "fy" << K(1, 1);
    fs << "cx" << K(0, 2);
    fs << "cy" << K(1, 2);

    fs << "k1" << distCoeffs[0];
    fs << "k2" << distCoeffs[1];
    fs << "p1" << distCoeffs[2];
    fs << "p2" << distCoeffs[3];
    fs << "k3" << distCoeffs[4];

    fs << "Image size" << frameSize;

    fs.release();

    std::cout << "Camera settings saved to \"" << filename << "\"\n";
    return EXIT_SUCCESS;

} catch (const cv::Exception& e) {
    std::cerr << "OpenCV error: " << e.what() << '\n';
    return EXIT_FAILURE;
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
getOptions(int argc, char* argv[], int& rows, int& cols, double& squareSize, double& tol, std::string& filename)
{
    if (argc != 5 && argc != 6) {
        return false;
    }

    if (!parseInt(argv[1], rows) || !parseInt(argv[2], cols)) {
        return false;
    }

    if (!parsePositiveDouble(argv[3], squareSize)) {
        return false;
    }

    if (argc == 6 && !parsePositiveDouble(argv[4], tol)) {
        return false;
    }

    filename = argv[argc - 1];
    filename += ".json";

    return true;
}

void
printUsage(const char* execName)
{
    std::cerr << "Usage: " << execName << " rows cols square_size [tol] filename\n"
              << "  rows, cols: Number of squares of the calibration pattern\n"
              << "  square_size: Size of the square, in cm\n"
              << "  tol: Per-view RMS rejection threshold (defaults to 1.5 pixels)\n"
              << "  filename: Output name, \".json\" is appended\n"
              << "  -h, --help: Show this message\n";
}

// TODO: This can be reused in MonoSLAM initialization
std::vector<cv::Point3f>
build3DCoordinates(const cv::Size& patternSize, double squareSize)
{
    std::vector<cv::Point3f> points3D;
    points3D.reserve(static_cast<std::size_t>(patternSize.width * patternSize.height));

    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            points3D.emplace_back(static_cast<float>(squareSize * i), static_cast<float>(squareSize * j), 0.0f);
        }
    }

    return points3D;
}
