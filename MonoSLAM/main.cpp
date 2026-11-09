/*
 * MonoSLAM: monocular EKF SLAM, initialized from a chessboard and then run on the natural features of the scene.
 * Every frame goes through the same steps:
 *
 *   1. Predict the camera with the motion model, and the image position of every feature, its seed, with the optical
 *      flow (tracking.h).
 *   2. After a weak frame, look for the map where the keyframes that look like the frame saw it, or else anywhere
 *      in the frame by its appearance, and move the camera to the pose found (relocalization.h).
 *   3. Search the features in view around their seeds, and those the camera comes back to around their prediction,
 *      and update the filter with the matches that 1-point RANSAC accepts (tracking.h, ekf.h).
 *   4. Replace the seeds with the projections of the features from the updated state.
 *   5. Keep the map: remove the features that are rarely matched, and add new ones where the frame has none (mapper.h).
 *
 * The filter works on 320 x 240 frames, which the camera gives at about 100 Hz with a fixed exposure, and the window
 * shows them enlarged. Tracking runs in its own thread, and the main thread draws and shows the frames, the last one it
 * is given, so that the window never slows tracking down (display.h).
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
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "display.h"
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
constexpr double imagesFps = 30.0;        // Assumed when the images have no timestamps
constexpr double linearAccStd = 1000.0;   // Linear acceleration standard deviation, in cm/s^2
constexpr double angularAccStd = 15.0;    // Angular acceleration standard deviation, in rad/s^2
constexpr double linearDecayTime = 0.3;   // Time constant of the decay of the linear velocity, in s
constexpr double angularDecayTime = 0.2;  // Time constant of the decay of the angular velocity, in s

// The filter works on frames of this size, for which every length in pixels in the code is chosen
const cv::Size frameSize(320, 240);

// Measurements
constexpr double pixelStd = 1.5;              // Standard deviation of a pixel measurement
constexpr double maxInlierDistance = 3.0;     // Pixels, for 1-point RANSAC
constexpr double maxNIS = 9.210340371976184;  // Chi-squared, 2 degrees of freedom, 99%

constexpr std::size_t minConstrained = 4;  // Matches needed to constrain the pose well

constexpr double goodAlignmentVariance = 0.04;  // Largest of a match aligned well, in pixels^2: 1 in 10 is worse

/*
 * Returns the noise of a match. It is the pixel noise, enlarged in the directions in which the alignment of its patch
 * was worse than a good one: along an edge, or where the patch does not look like the frame. It is never smaller, since
 * the pixel noise also covers what the alignment does not see.
 *
 * alignment    Covariance of the alignment of the patch
 */
cv::Matx22d
matchNoise(const cv::Matx22d& alignment)
{
    cv::Vec2d eigenvalues;
    cv::Matx22d eigenvectors;
    cv::eigen(alignment, eigenvalues, eigenvectors);

    const cv::Matx22d scale = cv::Matx22d::diag(cv::Vec2d(std::max(1.0, eigenvalues[0] / goodAlignmentVariance),
                                                          std::max(1.0, eigenvalues[1] / goodAlignmentVariance)));

    return eigenvectors.t() * scale * eigenvectors * (pixelStd * pixelStd);
}

struct Options
{
    cv::Size patternSize;     // Inner corners of the calibration pattern
    double squareSize = 0.0;  // Side of a square of the calibration pattern, in cm
    std::string cameraFile;   // Camera intrinsics

    std::string device;                           // Camera device, if not replaying images
    double exposure = monoslam::defaultExposure;  // In milliseconds

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
  "       camera: Calibration file, \".json\" is appended\n"
  "  --device D    Camera device, such as /dev/video4 (required, unless --images)\n"
  "  --exposure E  Exposure time of the camera, in ms (default 5): fixed, so that the frame rate is too\n"
  "  --images D    Replay the sorted images of directory D instead of a camera\n"
  "  --frames N    Stop after N frames\n"
  "  --stream F    Write the state of every frame to F, for tools/viewer.py and evaluate_tracking.py\n"
  "  --annotate D  Save the annotated frames to directory D\n"
  "  --headless    Do not open any window\n"
  "  --timing      Report every second the frame interval, the decoding and processing times, the frames\n"
  "                dropped and those shown\n";

/*
 * Reports every 30 frames, with --timing, the mean and largest frame interval and processing time, the decoding time,
 * and the frames dropped: live, frames are dropped while one is processed.
 */
class TimingReport
{
  public:
    /*
     * Adds a frame, and reports if it is the last of a period.
     *
     * index          Frame index
     * interval       Time since the previous frame, in s
     * processing     Time it took to process, in s
     * decoding       Time it took to decode, in s
     * numShown       Frames shown so far, or nothing if no window shows them
     * numFeatures    Features in the map
     */
    void add(std::size_t index,
             double interval,
             double processing,
             double decoding,
             std::optional<std::size_t> numShown,
             std::size_t numFeatures);

  private:
    static constexpr std::size_t period_ = 30;  // Frames

    double sumInterval_ = 0.0;
    double maxInterval_ = 0.0;
    double sumProcessing_ = 0.0;
    double maxProcessing_ = 0.0;
    double sumDecoding_ = 0.0;
    std::size_t numDropped_ = 0;
    std::size_t previousShown_ = 0;

    // The frame period of the camera, to count the frames dropped: half the shortest two intervals in a row, since some
    // cameras alternate a short and a long one (8 and 12 ms at 100 Hz)
    double framePeriod_ = HUGE_VAL;
    double previousInterval_ = HUGE_VAL;
};

/*
 * Parses the command line. Returns nothing if it is not valid.
 */
std::optional<Options>
parseOptions(int argc, char* argv[]);

/*
 * Initializes from the pattern and tracks until the input ends, or the display is stopped. Returns the exit status.
 *
 * options    Command line
 * camera     Camera model
 * source     Input, open
 * display    Where to post the frames to show, nullptr when headless
 */
int
track(const Options& options,
      const monoslam::CameraModel& camera,
      monoslam::FrameSource& source,
      monoslam::Display* display);

/*
 * Shows the frames posted to the display until it is stopped, and stops it if the user presses escape. Must run
 * in the main thread, as OpenCV's GUI needs.
 */
void
show(monoslam::Display& display);

/*
 * Saves an annotated frame to the directory of --annotate.
 */
void
save(const cv::Mat3b& image, std::size_t index, const Options& options);

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

    if (camera.frameSize() != frameSize) {
        std::cerr << options.cameraFile << " is a calibration for " << camera.frameSize().width << " x "
                  << camera.frameSize().height << " frames, and monoslam works on " << frameSize.width << " x "
                  << frameSize.height << ": calibrate again\n";
        return EXIT_FAILURE;
    }

    // The work is split into very small patches: threads cost more than they save
    cv::setNumThreads(1);

    // INPUT //

    monoslam::FrameSource source;

    if (options.imagesDir.empty() ? !source.openCamera(options.device, camera.frameSize(), options.exposure)
                                  : !source.openImages(options.imagesDir, camera.frameSize(), imagesFps)) {
        std::cerr << "Could not open the input\n";
        return EXIT_FAILURE;
    }

    if (options.headless) {
        return track(options, camera, source, nullptr);
    }

    monoslam::Display display;
    int status = EXIT_SUCCESS;

    std::thread tracking([&] {
        status = track(options, camera, source, &display);
        display.stop();
    });

    show(display);
    tracking.join();

    return status;
}

namespace {

int
track(const Options& options,
      const monoslam::CameraModel& camera,
      monoslam::FrameSource& source,
      monoslam::Display* display)
{
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

        if (display != nullptr) {
            if (display->stopped()) {
                return EXIT_SUCCESS;
            }

            display->post({ gray, {} });
        }
    } while (
      !monoslam::initializeFromPattern(gray, options.patternSize, options.squareSize, camera, x, R, P, features));

    std::cout << "Initialization succeeded\n";

    const monoslam::MotionModel motion{ linearAccStd, angularAccStd, linearDecayTime, angularDecayTime };
    const monoslam::MeasurementModel measurement{ pixelStd, maxInlierDistance, maxNIS };
    monoslam::EKF ekf(motion, measurement, camera, x, R, P);

    std::vector<cv::Point2f> initialPositions;

    for (const monoslam::Feature& feature : features) {
        initialPositions.emplace_back(feature.position);
    }

    monoslam::FlowTracker tracker(gray, initialPositions);
    monoslam::Mapper mapper(features.size());
    monoslam::Keyframes keyframes;

    monoslam::StateStream stream(camera);

    if (!options.streamFile.empty() && !stream.open(options.streamFile)) {
        std::cerr << "Could not open " << options.streamFile << '\n';
        return EXIT_FAILURE;
    }

    // TRACKING //

    bool weak = false;  // Whether too few of the features searched were used, which can mean the state is wrong
    double previousTime = source.time();

    TimingReport timing;

    while (source.read(gray)) {
        const std::size_t index = source.index();

        const auto processingStart = std::chrono::steady_clock::now();

        // 1. Predict the state, and the position of the features in the image. Live, frames are dropped while one
        //    is processed, so the interval is the real one
        const double dt = source.time() > previousTime ? source.time() - previousTime : 1.0 / imagesFps;
        previousTime = source.time();

        ekf.predict(dt);

        std::vector<cv::Point2f>& seeds = tracker.predict(gray);

        // 2. After a weak frame the state may be wrong. The features are searched where the keyframes that look like
        //    the frame saw them, or else anywhere in the frame by their descriptors, and the camera is reset
        //    to the pose they give
        if (weak) {
            std::optional<monoslam::Relocalization> found = keyframes.match(gray, features, camera, ekf);

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

        // 3. Search the features in view around their seeds, and those the camera comes back to around
        //    their prediction, and let the filter decide which matches to use
        const monoslam::Observation observation = ekf.observe();
        const std::vector<std::size_t>& searched = observation.features;

        std::vector<cv::Matx22d> alignments;
        const std::vector<std::optional<cv::Point2d>> matches =
          monoslam::matchFeatures(gray, features, camera, ekf, observation, seeds, alignments);

        std::vector<cv::Matx22d> noises;

        for (const cv::Matx22d& alignment : alignments) {
            noises.push_back(matchNoise(alignment));
        }

        std::vector<std::size_t> used = ekf.update(observation, matches, noises);

        // The matches to draw, for the window, --annotate or the stream
        const bool shown = display != nullptr || !options.annotateDir.empty();
        monoslam::Snapshot snapshot;

        if (shown || stream.isOpen()) {
            snapshot = { gray, monoslam::usedMatches(observation, seeds, matches, used) };
        }

        const std::size_t numUsed = used.size();

        // 4. If the state is well constrained, the seeds are replaced by the projections of the features. Otherwise
        //    they keep the positions given by the optical flow
        const std::vector<cv::Point2f> projections = ekf.project(seeds);

        if (numUsed >= minConstrained) {
            tracker.correct(projections);
        }

        if (stream.isOpen()) {
            stream.write(index, source.time(), gray, ekf, features, searched, used, snapshot.matches);
        }

        // 5. Keep the map: remove the features rarely matched, and add new ones where too few are in view.
        //    Not while most of the features searched fail, though: the state may be wrong, the failures would then
        //    count against the features, and new ones would be placed in the wrong position. Few features in view,
        //    with most of them found, means a new part of the scene instead, which needs new features: without them
        //    tracking would stay weak and the map frozen indefinitely
        weak = numUsed < minConstrained || 2 * numUsed < searched.size();

        if (2 * numUsed >= searched.size()) {
            mapper.recordMatches(features, searched, used);
            mapper.recordAppearance(gray, features, used, projections);
            mapper.removeUnreliable(ekf, features, tracker);
            mapper.addFeatures(gray, camera, ekf, features, tracker);
        }

        if (!weak) {
            keyframes.add(gray, ekf.pose(), numUsed);
        }

        if (options.timing) {
            const double processing =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - processingStart).count();
            const std::optional<std::size_t> numShown =
              display != nullptr ? std::optional<std::size_t>(display->numShown()) : std::nullopt;

            timing.add(index, dt, processing, source.decodeTime(), numShown, features.size());
        }

        if (shown) {
            if (!options.annotateDir.empty()) {
                save(monoslam::draw(snapshot), index, options);
            }

            if (display != nullptr) {
                display->post(std::move(snapshot));
            }
        }

        if ((display != nullptr && display->stopped()) || (options.maxFrames > 0 && index + 1 >= options.maxFrames)) {
            break;
        }
    }

    return EXIT_SUCCESS;
}

void
TimingReport::add(std::size_t index,
                  double interval,
                  double processing,
                  double decoding,
                  std::optional<std::size_t> numShown,
                  std::size_t numFeatures)
{
    sumInterval_ += interval;
    maxInterval_ = std::max(maxInterval_, interval);
    sumProcessing_ += processing;
    maxProcessing_ = std::max(maxProcessing_, processing);
    sumDecoding_ += decoding;
    framePeriod_ = std::min(framePeriod_, 0.5 * (previousInterval_ + interval));
    previousInterval_ = interval;
    numDropped_ += static_cast<std::size_t>(std::max(0L, std::lround(interval / framePeriod_) - 1));

    if (index % period_ != 0) {
        return;
    }

    const auto milliseconds = [](double seconds) { return std::lround(1000.0 * seconds); };

    std::cout << "Frame " << index << ": interval " << milliseconds(sumInterval_ / period_) << " ms (max "
              << milliseconds(maxInterval_) << "), processing " << milliseconds(sumProcessing_ / period_) << " ms (max "
              << milliseconds(maxProcessing_) << "), decoding " << milliseconds(sumDecoding_ / period_) << " ms, "
              << numDropped_ << " frames dropped";

    if (numShown.has_value()) {
        std::cout << ", " << numShown.value() - previousShown_ << " shown";
        previousShown_ = numShown.value();
    }

    std::cout << ", " << numFeatures << " features\n";

    sumInterval_ = maxInterval_ = sumProcessing_ = maxProcessing_ = sumDecoding_ = 0.0;
    numDropped_ = 0;
}

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

        if (option == "--headless") {
            options.headless = true;

        } else if (option == "--timing") {
            options.timing = true;

        } else if (option == "--device" && hasValue) {
            options.device = argv[++i];

        } else if (option == "--exposure" && hasValue && parse(argv[i + 1], options.exposure)) {
            ++i;

        } else if (option == "--images" && hasValue) {
            options.imagesDir = argv[++i];

        } else if (option == "--frames" && hasValue && parse(argv[i + 1], options.maxFrames)) {
            ++i;

        } else if (option == "--stream" && hasValue) {
            options.streamFile = argv[++i];

        } else if (option == "--annotate" && hasValue) {
            options.annotateDir = argv[++i];

        } else {
            return std::nullopt;
        }
    }

    // Exactly one input
    if (options.device.empty() == options.imagesDir.empty()) {
        return std::nullopt;
    }

    return options;
}

void
show(monoslam::Display& display)
{
    constexpr int minWidth = 640;  // Of the frames shown, enlarged if smaller

    cv::namedWindow("MonoSLAM", cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_NORMAL);

    while (!display.stopped()) {
        if (std::optional<monoslam::Snapshot> snapshot = display.take()) {
            // Enlarged before drawing, so that the drawings are as sharp as the screen
            const int zoom = std::max(1, minWidth / snapshot->frame.cols);

            if (zoom > 1) {
                cv::resize(snapshot->frame, snapshot->frame, cv::Size(), zoom, zoom, cv::INTER_LINEAR);
                snapshot->scale *= zoom;
            }

            cv::imshow("MonoSLAM", monoslam::draw(snapshot.value()));
        }

        // Also lets the window handle its events while no frame comes
        if (cv::waitKey(5) == 27) {
            display.stop();
        }
    }
}

void
save(const cv::Mat3b& image, std::size_t index, const Options& options)
{
    std::filesystem::create_directories(options.annotateDir);

    std::ostringstream name;
    name << options.annotateDir << "/frame_" << std::setw(6) << std::setfill('0') << index << ".png";

    cv::imwrite(name.str(), image);
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
