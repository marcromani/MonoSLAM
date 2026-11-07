/*
 * Calibrates the camera with a chessboard: it captures a view of the pattern every two seconds
 * until Escape is pressed, fits the pinhole model and the lens distortion to all of them, drops
 * the views that the fit explains badly, fits again, and saves the result for monoslam. The fit is
 * on the largest frames the camera gives of the same view as monoslam's, where the corners are the
 * most precise, and the model is saved for monoslam's frames, scaled to their size.
 */

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

namespace {

const cv::Size frameSize(320, 240);  // Of the frames monoslam works on
// The largest mode of the camera with the same view as frameSize, only larger. Not every mode of
// the same shape is one: this camera's 1024 x 768 sees a view 12% wider and taller than its 320 x 240
const cv::Size captureSize(640, 480);

constexpr double autoExposure = 3.0;  // V4L2_EXPOSURE_APERTURE_PRIORITY, which monoslam turns off

constexpr auto samplePeriod = std::chrono::seconds(2);  // Between captured views, so that they differ

struct Options
{
    cv::Size patternSize;     // Inner corners of the pattern
    double squareSize = 0.0;  // Side of a square, in cm
    std::string cameraFile;   // Calibration file to write
    std::string device;       // Camera device
    double tolerance = 1.5;   // Largest RMS reprojection error of a view that is kept, in pixels of the camera
};

constexpr const char* usage =
  "Usage: calibration rows cols square_size camera [options]\n"
  "   rows, cols: Number of squares of the calibration pattern\n"
  "  square_size: Side of a square, in cm\n"
  "       camera: Calibration file to write, \".json\" is appended\n"
  "  --device D     Camera device, such as /dev/video4 (required)\n"
  "  --tolerance T  Largest RMS reprojection error of a view that is kept, in pixels of the camera frames\n"
  "                 (default 1.5)\n";

std::optional<Options>
parseOptions(int argc, char* argv[])
{
    const auto parse = [](std::string_view text, auto& value) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc() && end == text.data() + text.size() && value > 0;
    };

    Options options;
    int rows = 0, cols = 0;

    if (argc < 5 || !parse(argv[1], rows) || !parse(argv[2], cols) || !parse(argv[3], options.squareSize)) {
        return std::nullopt;
    }

    options.patternSize = cv::Size(cols - 1, rows - 1);
    options.cameraFile = std::string(argv[4]) + ".json";

    for (int i = 5; i < argc; ++i) {
        const std::string_view option(argv[i]);
        const bool hasValue = i + 1 < argc;

        if (option == "--device" && hasValue) {
            options.device = argv[++i];

        } else if (option == "--tolerance" && hasValue && parse(argv[i + 1], options.tolerance)) {
            ++i;

        } else {
            return std::nullopt;
        }
    }

    if (options.device.empty()) {
        return std::nullopt;
    }

    return options;
}

}

int
main(int argc, char* argv[])
try {
    const std::optional<Options> parsed = parseOptions(argc, argv);

    if (!parsed.has_value()) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    const Options& options = parsed.value();

    cv::VideoCapture device(options.device, cv::CAP_V4L2);

    if (!device.isOpened()) {
        std::cerr << "Could not open " << options.device << '\n';
        return EXIT_FAILURE;
    }

    device.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    device.set(cv::CAP_PROP_FRAME_WIDTH, captureSize.width);
    device.set(cv::CAP_PROP_FRAME_HEIGHT, captureSize.height);
    device.set(cv::CAP_PROP_AUTO_EXPOSURE, autoExposure);  // The views are still: no need for a short one

    // Another mode, of another shape, would not scale to frameSize
    const cv::Size imageSize(static_cast<int>(device.get(cv::CAP_PROP_FRAME_WIDTH)),
                             static_cast<int>(device.get(cv::CAP_PROP_FRAME_HEIGHT)));

    if (imageSize != captureSize) {
        std::cerr << options.device << " gives " << imageSize.width << " x " << imageSize.height
                  << " frames instead of " << captureSize.width << " x " << captureSize.height << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Press Escape to finish capturing views\n";

    // The camera on the left, and the last view captured on the right, in gray with the pattern
    // found in color
    cv::Mat3b display(imageSize.height, 2 * imageSize.width, cv::Vec3b::zeros());
    cv::Mat3b live = display(cv::Rect(0, 0, imageSize.width, imageSize.height));
    cv::Mat3b captured = display(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height));

    cv::namedWindow("Camera calibration", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);

    std::vector<std::vector<cv::Point2f>> views;  // Corners of the pattern in every view
    auto lastCapture = std::chrono::steady_clock::now();

    while (cv::waitKey(1) != 27) {
        cv::Mat3b frame;

        if (!device.read(frame) || frame.empty()) {
            std::cerr << "\nCould not read a camera frame\n";
            return EXIT_FAILURE;
        }

        frame.copyTo(live);

        std::vector<cv::Point2f> corners;

        if (std::chrono::steady_clock::now() - lastCapture > samplePeriod &&
            cv::findChessboardCorners(frame,
                                      options.patternSize,
                                      corners,
                                      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                                        cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK)) {
            cv::Mat1b gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(30, 0.1));

            views.push_back(corners);
            lastCapture = std::chrono::steady_clock::now();

            cv::cvtColor(gray, captured, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(captured, options.patternSize, corners, true);

            std::cout << '\r' << views.size() << " views captured" << std::flush;
        }

        cv::imshow("Camera calibration", display);
    }

    cv::destroyAllWindows();
    std::cout << '\n';

    if (views.empty()) {
        std::cerr << "No views to calibrate with\n";
        return EXIT_FAILURE;
    }

    // The corners in the plane of the pattern, the same for every view
    std::vector<cv::Point3f> pattern;

    for (int i = 0; i < options.patternSize.height; ++i) {
        for (int j = 0; j < options.patternSize.width; ++j) {
            pattern.emplace_back(
              static_cast<float>(options.squareSize * i), static_cast<float>(options.squareSize * j), 0.0f);
        }
    }

    cv::Matx33d K;
    cv::Vec<double, 5> distCoeffs;
    std::vector<cv::Vec3d> rotations, translations;
    std::vector<double> viewErrors;

    double rms = cv::calibrateCamera(std::vector(views.size(), pattern),
                                     views,
                                     imageSize,
                                     K,
                                     distCoeffs,
                                     rotations,
                                     translations,
                                     cv::noArray(),
                                     cv::noArray(),
                                     viewErrors);

    // The views the first fit explains badly are probably wrong detections, or blurred
    std::vector<std::vector<cv::Point2f>> kept;

    for (std::size_t i = 0; i < views.size(); ++i) {
        if (viewErrors[i] < options.tolerance) {
            kept.push_back(views[i]);
        }
    }

    if (kept.empty()) {
        std::cerr << "No good views to calibrate with\n";
        return EXIT_FAILURE;
    }

    if (kept.size() < views.size()) {
        rms = cv::calibrateCamera(
          std::vector(kept.size(), pattern), kept, imageSize, K, distCoeffs, rotations, translations);
    }

    // For the frames monoslam works on: the center of a pixel scales with the frame, its corner does not
    const double sx = static_cast<double>(frameSize.width) / imageSize.width;
    const double sy = static_cast<double>(frameSize.height) / imageSize.height;

    K = cv::Matx33d(
      K(0, 0) * sx, 0.0, (K(0, 2) + 0.5) * sx - 0.5, 0.0, K(1, 1) * sy, (K(1, 2) + 0.5) * sy - 0.5, 0.0, 0.0, 1.0);

    std::cout << "Calibrated with " << kept.size() << " of " << views.size() << " views of " << imageSize.width << " x "
              << imageSize.height << ", RMS reprojection error " << rms << " pixels\n"
              << "For " << frameSize.width << " x " << frameSize.height << " frames:\n"
              << "fx = " << K(0, 0) << ", fy = " << K(1, 1) << ", cx = " << K(0, 2) << ", cy = " << K(1, 2) << '\n'
              << "k1 = " << distCoeffs[0] << ", k2 = " << distCoeffs[1] << ", p1 = " << distCoeffs[2]
              << ", p2 = " << distCoeffs[3] << ", k3 = " << distCoeffs[4] << '\n';

    cv::FileStorage file(options.cameraFile, cv::FileStorage::WRITE);

    if (!file.isOpened()) {
        std::cerr << "Could not open " << options.cameraFile << '\n';
        return EXIT_FAILURE;
    }

    file << "fx" << K(0, 0) << "fy" << K(1, 1) << "cx" << K(0, 2) << "cy" << K(1, 2);
    file << "k1" << distCoeffs[0] << "k2" << distCoeffs[1] << "p1" << distCoeffs[2] << "p2" << distCoeffs[3] << "k3"
         << distCoeffs[4];
    file << "Frame size" << frameSize;

    std::cout << "Saved to " << options.cameraFile << '\n';

    return EXIT_SUCCESS;

} catch (const cv::Exception& e) {
    std::cerr << "OpenCV error: " << e.what() << '\n';
    return EXIT_FAILURE;
}
