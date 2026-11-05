/*
 * record_sequence: records a sequence for monoslam to replay with --images. The frames are saved
 * as frame_000000.png, ... in grayscale, at the size of the calibration, and timestamps.csv has the
 * time at which the camera captured each of them, which is what the filter needs. The calibration is
 * copied next to them, as camera.json.
 *
 * With --frames or --seconds, recording starts right away and stops at the limit. Otherwise it
 * starts when Space is pressed in the preview window. Escape stops it in both cases.
 */

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "camera.h"
#include "frame_source.h"

namespace {

struct Options
{
    std::string cameraFile;   // Calibration, as the calibration program writes it
    std::string device;       // Camera device
    std::string directory;    // Where the sequence goes, empty or not yet created
    double fps = 30.0;        // Frame rate to ask the camera for
    long maxFrames = 0;       // Frames to record, 0 for no limit
    double maxSeconds = 0.0;  // Seconds to record, 0 for no limit
    bool headless = false;    // Without the preview window, only with a limit
};

constexpr const char* usage = "Usage: record_sequence camera device directory [options]\n"
                              "     camera: Calibration file, without \".json\"\n"
                              "     device: Camera device, such as /dev/video5\n"
                              "  directory: Where the sequence goes, empty or not yet created\n"
                              "  --fps F        Frame rate to ask the camera for (default 30)\n"
                              "  --frames N     Record N frames, starting right away\n"
                              "  --seconds S    Record S seconds, starting right away\n"
                              "  --headless     Without the preview window, only with --frames or --seconds\n";

/*
 * Parses the command line. Returns nothing if it is not valid.
 */
std::optional<Options>
parseOptions(int argc, char* argv[])
{
    const auto parse = [](std::string_view text, auto& value) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc() && end == text.data() + text.size() && value > 0;
    };

    if (argc < 4) {
        return std::nullopt;
    }

    Options options;
    options.cameraFile = std::string(argv[1]) + ".json";
    options.device = argv[2];
    options.directory = argv[3];

    for (int i = 4; i < argc; ++i) {
        const std::string_view option(argv[i]);
        const bool hasValue = i + 1 < argc;

        if (option == "--headless") {
            options.headless = true;

        } else if (option == "--fps" && hasValue && parse(argv[i + 1], options.fps)) {
            ++i;

        } else if (option == "--frames" && hasValue && parse(argv[i + 1], options.maxFrames)) {
            ++i;

        } else if (option == "--seconds" && hasValue && parse(argv[i + 1], options.maxSeconds)) {
            ++i;

        } else {
            return std::nullopt;
        }
    }

    // Without a window there is no way to start or stop recording
    if (options.headless && options.maxFrames == 0 && options.maxSeconds == 0.0) {
        return std::nullopt;
    }

    return options;
}

}

int
main(int argc, char* argv[])
{
    namespace fs = std::filesystem;

    const std::optional<Options> parsed = parseOptions(argc, argv);

    if (!parsed.has_value()) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    const Options& options = parsed.value();

    const std::optional<monoslam::CameraModel> camera = monoslam::CameraModel::load(options.cameraFile);

    if (!camera.has_value()) {
        std::cerr << "Could not read " << options.cameraFile << '\n';
        return EXIT_FAILURE;
    }

    monoslam::FrameSource source;

    if (!source.openCamera(options.device, camera->frameSize(), options.fps)) {
        std::cerr << "Could not open " << options.device << '\n';
        return EXIT_FAILURE;
    }

    // A sequence is never mixed with the frames of another one
    std::error_code error;

    if (fs::exists(options.directory) && !fs::is_empty(options.directory)) {
        std::cerr << options.directory << " is not empty\n";
        return EXIT_FAILURE;
    }

    if (fs::create_directories(options.directory, error);
        error || !fs::copy_file(options.cameraFile, fs::path(options.directory) / "camera.json", error)) {
        std::cerr << "Could not write to " << options.directory << ": " << error.message() << '\n';
        return EXIT_FAILURE;
    }

    std::ofstream timestamps(fs::path(options.directory) / "timestamps.csv");
    timestamps << "filename,capture_us\n";

    bool recording = options.maxFrames > 0 || options.maxSeconds > 0.0;
    double startTime = 0.0;
    long numFrames = 0;

    cv::Mat1b frame;

    while (source.read(frame)) {
        int key = -1;

        if (!options.headless) {
            cv::imshow(recording ? "Recording, Escape to stop" : "Space to record, Escape to quit", frame);
            key = cv::waitKey(1);
        }

        if (key == 27) {
            break;
        }

        if (!recording && key == ' ') {
            cv::destroyAllWindows();  // The title of the window changes
            recording = true;
        }

        if (!recording) {
            continue;
        }

        if (numFrames == 0) {
            startTime = source.time();
        }

        std::ostringstream name;
        name << "frame_" << std::setw(6) << std::setfill('0') << numFrames << ".png";

        if (!cv::imwrite((fs::path(options.directory) / name.str()).string(), frame)) {
            std::cerr << "Could not write " << name.str() << '\n';
            return EXIT_FAILURE;
        }

        timestamps << name.str() << ',' << std::llround(1e6 * source.time()) << '\n';
        ++numFrames;

        if ((options.maxFrames > 0 && numFrames >= options.maxFrames) ||
            (options.maxSeconds > 0.0 && source.time() - startTime >= options.maxSeconds)) {
            break;
        }
    }

    std::cout << "Recorded " << numFrames << " frames to " << options.directory << '\n';

    return numFrames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
