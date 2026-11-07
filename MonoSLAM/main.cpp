/*
 * MonoSLAM: monocular EKF SLAM, initialized from a chessboard and then run on the natural features
 * of the scene. Every frame goes through the same steps:
 *
 *   1. Predict the camera with the motion model, and the image position of every feature, its
 *      seed, with the optical flow (tracking.h).
 *   2. After a weak frame, look for the map anywhere in the frame by its appearance, and move the
 *      camera where it says (relocalization.h).
 *   3. Search the features in view around their seeds, and those the camera comes back to around
 *      their prediction, and update the filter with the matches that 1-point RANSAC accepts
 *      (tracking.h, ekf.h).
 *   4. Let the seeds follow the updated state.
 *   5. Keep the map: remove the features that are rarely matched, and add new ones where the frame
 *      has none (mapper.h).
 */

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "ekf.h"
#include "feature.h"
#include "frame_source.h"
#include "initialization.h"
#include "mapper.h"
#include "relocalization.h"
#include "state_stream.h"
#include "tracking.h"

namespace {

// Motion model
constexpr double FPS = 30.0;              // Assumed when the frames have no timestamps
constexpr double linearAccStd = 1000.0;   // Linear acceleration standard deviation, in cm/s^2
constexpr double angularAccStd = 15.0;    // Angular acceleration standard deviation, in rad/s^2
constexpr double linearDecayTime = 0.3;   // Time constant of the decay of the linear velocity, in s
constexpr double angularDecayTime = 0.2;  // Time constant of the decay of the angular velocity, in s

// Measurements
constexpr double pixelStd = 3.0;              // Standard deviation of a pixel measurement
constexpr double maxInlierDistance = 6.0;     // Pixels, for 1-point RANSAC
constexpr double maxNIS = 9.210340371976184;  // Chi-squared, 2 degrees of freedom, 99%

constexpr std::size_t minConstrained = 4;  // Measurements needed to fix a pose beyond doubt

struct Options
{
    cv::Size patternSize;     // Inner corners of the calibration pattern
    double squareSize = 0.0;  // Side of a square of the calibration pattern, in cm
    std::string cameraFile;   // Camera intrinsics
    std::string device = "/dev/video5";
    std::string imagesDir;      // Replay the images of this directory instead of using the camera
    std::string streamFile;     // Write the state of every frame here, for tools/viewer.py
    std::string annotateDir;    // Save the annotated frames here
    std::size_t maxFrames = 0;  // Stop after this frame, 0 to never stop
    bool headless = false;      // Do not open any window
    bool timing = false;        // Report the frame intervals, the processing time and the dropped frames
};

constexpr const char* usage =
  "Usage: monoslam rows cols square_size camera [options]\n"
  "   rows, cols: Number of squares of the calibration pattern\n"
  "  square_size: Side of a square, in cm\n"
  "       camera: Camera intrinsics file, \".json\" is appended\n"
  "  --device D    Camera device (default /dev/video5)\n"
  "  --images D    Replay the sorted images of directory D instead of a camera\n"
  "  --frames N    Stop after N frames\n"
  "  --stream F    Write the state of every frame to F, for tools/viewer.py and evaluate_tracking.py\n"
  "  --annotate D  Save the annotated frames to directory D\n"
  "  --headless    Do not open any window\n"
  "  --timing      Report every second the frame interval, processing time and dropped frames\n";

/*
 * Parses the command line. Returns nothing if it is not valid.
 */
std::optional<Options>
parseOptions(int argc, char* argv[]);

/*
 * Draws the features whose matches the filter used: the predicted measurement (cyan) with the 99%
 * ellipse of its innovation covariance, the seed (magenta), the match (amber), and the innovation
 * from the prediction to the match (white).
 *
 * image      Image to draw on
 * o          Features in view, and their predicted measurements
 * seeds      Seeds of all the features
 * matches    Pixel of each feature of o, if it was found
 * used       Features the filter used
 */
void
drawUsedMatches(cv::Mat3b& image,
                const monoslam::Observation& o,
                const std::vector<cv::Point2f>& seeds,
                const std::vector<std::optional<cv::Point2d>>& matches,
                const std::vector<std::size_t>& used);

/*
 * Shows a frame, unless running headless, and saves it if asked to. Returns false if the user
 * pressed escape.
 */
bool
showFrame(const cv::Mat& frame, std::size_t index, const Options& options);

}

int
main(int argc, char* argv[])
{
    const std::optional<Options> parsed = parseOptions(argc, argv);

    if (!parsed.has_value()) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    const Options& options = parsed.value();

    const std::optional<monoslam::CameraModel> calibration = monoslam::CameraModel::load(options.cameraFile);

    if (!calibration.has_value()) {
        std::cerr << "Could not read " << options.cameraFile << '\n';
        return EXIT_FAILURE;
    }

    const monoslam::CameraModel& camera = calibration.value();

    // The work is split into very small patches: threads cost more than they save
    cv::setNumThreads(1);

    // INPUT //

    monoslam::FrameSource source;

    if (options.imagesDir.empty() ? !source.openCamera(options.device, camera.frameSize(), FPS)
                                  : !source.openImages(options.imagesDir, camera.frameSize(), FPS)) {
        std::cerr << "Could not open the input\n";
        return EXIT_FAILURE;
    }

    if (!options.headless) {
        cv::namedWindow("MonoSLAM", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);
    }

    // INITIALIZATION //

    cv::Mat1b gray;
    cv::Mat1d x;
    cv::Matx33d R;
    cv::Mat1d P;

    std::vector<monoslam::Feature> features;

    do {
        if (!source.read(gray)) {
            std::cerr << "The pattern was not found in the sequence\n";
            return EXIT_FAILURE;
        }

        if (!options.headless) {
            cv::imshow("MonoSLAM", gray);

            if (cv::waitKey(1) == 27) {
                return EXIT_SUCCESS;
            }
        }
    } while (
      !monoslam::initializeFromPattern(gray, options.patternSize, options.squareSize, camera, x, R, P, features));

    std::cout << "Initialization succeeded\n";

    const monoslam::MotionModel motion{ linearAccStd, angularAccStd, linearDecayTime, angularDecayTime };
    monoslam::EKF ekf(motion, camera, pixelStd, x, R, P);

    std::vector<cv::Point2f> initialPositions;

    for (const monoslam::Feature& feature : features) {
        initialPositions.emplace_back(feature.position);
    }

    monoslam::FlowTracker tracker(gray, initialPositions);
    monoslam::Mapper mapper(features.size());
    monoslam::Keyframes keyframes;

    monoslam::StateStream stream;

    if (!options.streamFile.empty() && !stream.open(options.streamFile)) {
        std::cerr << "Could not open " << options.streamFile << '\n';
        return EXIT_FAILURE;
    }

    // TRACKING //

    bool weak = false;  // Whether too few of the features searched were used, which can mean the state is wrong
    double previousTime = source.time();

    // Live timing, reported every second with --timing
    constexpr std::size_t timingPeriod = 30;  // Frames
    double sumInterval = 0.0, maxInterval = 0.0, sumProcessing = 0.0, maxProcessing = 0.0;
    std::size_t numDropped = 0;

    while (source.read(gray)) {
        const std::size_t index = source.index();

        cv::Mat3b frame;
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);  // To draw on

        const auto processingStart = std::chrono::steady_clock::now();

        // 1. Predict the state, and the position of the features in the image. Live, frames are
        //    dropped while one is processed, so the interval is the real one
        const double dt = source.time() > previousTime ? source.time() - previousTime : 1.0 / FPS;
        previousTime = source.time();

        ekf.predict(dt);

        std::vector<cv::Point2f>& seeds = tracker.predict(gray);

        // 2. After a weak frame the state may be wrong: the features are looked for where the
        //    keyframes that look like the frame saw them, or else anywhere in the frame by their
        //    descriptors, and the camera is moved where they say
        if (weak) {
            std::optional<monoslam::GlobalMatch> found = keyframes.match(gray, features, camera, ekf);

            if (!found.has_value()) {
                found = monoslam::matchGlobally(gray, features, camera, ekf);
            }

            if (found.has_value()) {
                ekf.resetCamera(found->r, found->R);

                seeds = ekf.project(seeds);

                for (std::size_t k = 0; k < found->features.size(); ++k) {
                    seeds[found->features[k]] = cv::Point2f(found->pixels[k]);
                }
            }
        }

        // 3. Search the features in view around their seeds, and those the camera comes back to
        //    around their prediction, and let the filter decide which matches to use
        const monoslam::Observation observation = ekf.observe();
        const std::vector<std::size_t>& searched = observation.features;

        const std::vector<std::optional<cv::Point2d>> matches =
          monoslam::matchFeatures(gray, features, camera, ekf, observation, seeds);

        std::vector<std::size_t> used = ekf.update(observation, matches, maxInlierDistance, maxNIS);

        drawUsedMatches(frame, observation, seeds, matches, used);

        const std::size_t numMeasured = used.size();

        // 4. The seeds follow the state if it is well constrained, and the images otherwise
        const std::vector<cv::Point2f> projections = ekf.project(seeds);

        if (numMeasured >= minConstrained) {
            tracker.correct(projections);
        }

        if (stream.isOpen()) {
            stream.write(index, ekf, features, searched, used);
        }

        // 5. Keep the map: remove the features rarely matched, and add new ones where too few are in
        //    view. Not while most of the features searched fail, though: the state may be wrong, and
        //    then the features would be blamed for it, and the new ones placed where they are not.
        //    Few features in view, most of them found, is a new part of the scene instead, which
        //    needs new features: without them tracking would stay weak, and the map frozen, for good
        weak = numMeasured < minConstrained || 2 * numMeasured < searched.size();

        if (2 * numMeasured >= searched.size()) {
            mapper.recordMatches(features, searched, used);
            mapper.recordAppearance(gray, features, used, projections);
            mapper.removeUnreliable(ekf, features, tracker);
            mapper.addFeatures(gray, camera, ekf, features, tracker);
        }

        if (!weak) {
            keyframes.add(gray, ekf.pose(), numMeasured);
        }

        if (options.timing) {
            const double processing =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - processingStart).count();

            sumInterval += dt;
            maxInterval = std::max(maxInterval, dt);
            sumProcessing += processing;
            maxProcessing = std::max(maxProcessing, processing);
            numDropped += static_cast<std::size_t>(std::max(0L, std::lround(dt * FPS) - 1));

            if (index % timingPeriod == 0) {
                std::cout << "Frame " << index << ": interval " << std::lround(1000.0 * sumInterval / timingPeriod)
                          << " ms (max " << std::lround(1000.0 * maxInterval) << "), processing "
                          << std::lround(1000.0 * sumProcessing / timingPeriod) << " ms (max "
                          << std::lround(1000.0 * maxProcessing) << "), " << numDropped << " frames dropped, "
                          << features.size() << " features\n";

                sumInterval = maxInterval = sumProcessing = maxProcessing = 0.0;
                numDropped = 0;
            }
        }

        if (!showFrame(frame, index, options)) {
            break;
        }

        if (options.maxFrames > 0 && index + 1 >= options.maxFrames) {
            break;
        }
    }

    return EXIT_SUCCESS;
}

namespace {

std::optional<Options>
parseOptions(int argc, char* argv[])
{
    const auto parse = [](std::string_view text, auto& value) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc() && end == text.data() + text.size() && value > 0;
    };

    Options options;
    int rows = 0, cols = 0, maxFrames = 0;

    if (argc < 5 || !parse(argv[1], rows) || !parse(argv[2], cols) || !parse(argv[3], options.squareSize)) {
        return std::nullopt;
    }

    options.patternSize = cv::Size(cols - 1, rows - 1);
    options.cameraFile = std::string(argv[4]) + ".json";

    for (int i = 5; i < argc; ++i) {
        const std::string_view option(argv[i]);

        if (option == "--headless") {
            options.headless = true;

        } else if (option == "--timing") {
            options.timing = true;

        } else if (i + 1 == argc) {
            return std::nullopt;

        } else if (option == "--frames") {
            if (!parse(argv[++i], maxFrames)) {
                return std::nullopt;
            }

            options.maxFrames = static_cast<std::size_t>(maxFrames);

        } else if (option == "--device") {
            options.device = argv[++i];

        } else if (option == "--images") {
            options.imagesDir = argv[++i];

        } else if (option == "--stream") {
            options.streamFile = argv[++i];

        } else if (option == "--annotate") {
            options.annotateDir = argv[++i];

        } else {
            return std::nullopt;
        }
    }

    return options;
}

void
drawUsedMatches(cv::Mat3b& image,
                const monoslam::Observation& o,
                const std::vector<cv::Point2f>& seeds,
                const std::vector<std::optional<cv::Point2d>>& matches,
                const std::vector<std::size_t>& used)
{
    const cv::Scalar cyan(230.0, 200.0, 60.0);  // BGR
    const cv::Scalar amber(0.0, 170.0, 255.0);
    const cv::Scalar magenta(200.0, 60.0, 230.0);
    const cv::Scalar dark(30.0, 30.0, 30.0);

    constexpr double numSigmas = 3.035;  // The 99% region of a 2D Gaussian
    constexpr double fillOpacity = 0.2;

    struct Drawn
    {
        cv::RotatedRect ellipse;
        cv::Point2d prediction;
        cv::Point2d seed;
        cv::Point2d match;
    };

    std::vector<Drawn> drawn;

    for (std::size_t k = 0; k < o.features.size(); ++k) {
        const std::size_t feature = o.features[k];

        if (!matches[k].has_value() || std::find(used.begin(), used.end(), feature) == used.end()) {
            continue;
        }

        const int row = static_cast<int>(2 * k);
        const cv::Point2d prediction(o.h(row), o.h(row + 1));

        // The axes of the ellipse are the eigenvectors of the innovation covariance
        cv::Vec2d eigenvalues;
        cv::Matx22d eigenvectors;
        cv::eigen(cv::Matx22d(o.S(cv::Rect(row, row, 2, 2))), eigenvalues, eigenvectors);

        const cv::Size2d size(2.0 * numSigmas * std::sqrt(eigenvalues[0]), 2.0 * numSigmas * std::sqrt(eigenvalues[1]));
        const double angle = std::atan2(eigenvectors(0, 1), eigenvectors(0, 0)) * 180.0 / CV_PI;

        drawn.push_back({ cv::RotatedRect(prediction, size, static_cast<float>(angle)),
                          prediction,
                          seeds[feature],
                          matches[k].value() });
    }

    // Ellipses much larger than the image say nothing, and are expensive to draw
    const auto drawable = [&](const cv::RotatedRect& e) {
        return std::isfinite(e.size.width) && e.size.width < 20.0F * static_cast<float>(image.cols);
    };

    // The regions, translucent, so that the image shows through them
    cv::Mat3b filled = image.clone();

    for (const Drawn& d : drawn) {
        if (drawable(d.ellipse)) {
            cv::ellipse(filled, d.ellipse, cyan, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::addWeighted(filled, fillOpacity, image, 1.0 - fillOpacity, 0.0, image);

    // A dot with a dark rim, visible on any background
    const auto dot = [&](const cv::Point2d& p, const cv::Scalar& color) {
        cv::circle(image, p, 4, dark, cv::FILLED, cv::LINE_AA);
        cv::circle(image, p, 3, color, cv::FILLED, cv::LINE_AA);
    };

    for (const Drawn& d : drawn) {
        if (drawable(d.ellipse)) {
            cv::ellipse(image, d.ellipse, cyan, 1, cv::LINE_AA);
        }

        // The innovation, from the prediction to the match
        cv::line(image, d.prediction, d.match, cv::Scalar::all(255.0), 1, cv::LINE_AA);

        dot(d.seed, magenta);
        dot(d.prediction, cyan);
        dot(d.match, amber);
    }
}

bool
showFrame(const cv::Mat& frame, std::size_t index, const Options& options)
{
    if (!options.annotateDir.empty()) {
        std::filesystem::create_directories(options.annotateDir);

        std::ostringstream name;
        name << options.annotateDir << "/frame_" << std::setw(6) << std::setfill('0') << index << ".png";

        cv::imwrite(name.str(), frame);
    }

    if (!options.headless) {
        cv::imshow("MonoSLAM", frame);

        return cv::waitKey(1) != 27;
    }

    return true;
}

}

#if defined(MONOSLAM_SANITIZERS_ENABLED)
#    include <sanitizer/lsan_interface.h>

// OpenCV's GUI keeps an object alive until the process ends, which the leak sanitizer reports
extern "C" __attribute__((used)) const char*
__lsan_default_suppressions()
{
    return "leak:GuiReceiver::GuiReceiver\n";
}
#endif
