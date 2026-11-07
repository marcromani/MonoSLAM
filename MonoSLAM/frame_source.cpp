#include "frame_source.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

bool
monoslam::FrameSource::openCamera(const std::string& device, const cv::Size& frameSize, double fps)
{
    frameSize_ = frameSize;

    device_.open(device, cv::CAP_V4L2);

    if (!device_.isOpened()) {
        return false;
    }

    device_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    device_.set(cv::CAP_PROP_FPS, fps);
    device_.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
    device_.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
    device_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Do not fall behind if an iteration takes too long

    return true;
}

bool
monoslam::FrameSource::openImages(const std::string& directory, const cv::Size& frameSize, double fps)
{
    namespace fs = std::filesystem;

    frameSize_ = frameSize;
    files_.clear();

    if (!fs::is_directory(directory)) {
        return false;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        const std::string extension = entry.path().extension().string();

        if (entry.is_regular_file() && (extension == ".png" ||
                                        extension == ".jpg" ||
                                        extension == ".jpeg" ||
                                        extension == ".bmp" ||
                                        extension == ".pgm" ||
                                        extension == ".ppm")) {
            files_.push_back(entry.path().string());
        }
    }

    std::sort(files_.begin(), files_.end());

    // The capture times, if the file has a column capture_us (older recordings only have the time
    // at which the recorder read the frames, which is no use)
    std::map<std::string, double> captureTimes;
    std::ifstream file(fs::path(directory) / "timestamps.csv");
    std::string line;

    if (std::getline(file, line)) {
        std::vector<std::string> header;
        std::istringstream names(line);

        for (std::string name; std::getline(names, name, ',');) {
            header.push_back(name);
        }

        const auto filenameColumn = std::find(header.begin(), header.end(), "filename");
        const auto captureColumn = std::find(header.begin(), header.end(), "capture_us");

        if (filenameColumn != header.end() && captureColumn != header.end()) {
            while (std::getline(file, line)) {
                std::vector<std::string> fields;
                std::istringstream values(line);

                for (std::string value; std::getline(values, value, ',');) {
                    fields.push_back(value);
                }

                if (fields.size() == header.size()) {
                    captureTimes[fields[static_cast<std::size_t>(filenameColumn - header.begin())]] =
                      1e-6 * std::stod(fields[static_cast<std::size_t>(captureColumn - header.begin())]);
                }
            }
        }
    }

    times_.clear();

    for (const std::string& image : files_) {
        const auto found = captureTimes.find(fs::path(image).filename().string());

        if (found == captureTimes.end()) {
            times_.clear();  // All or nothing
            break;
        }

        times_.push_back(found->second);
    }

    if (times_.empty()) {
        for (std::size_t i = 0; i < files_.size(); ++i) {
            times_.push_back(static_cast<double>(i) / fps);
        }
    }

    return !files_.empty();
}

bool
monoslam::FrameSource::read(cv::Mat1b& frame)
{
    cv::Mat3b color;

    auto start = std::chrono::steady_clock::now();

    if (files_.empty()) {
        // The wait for the next frame is in grab, the decoding in retrieve
        while (!device_.grab()) {
        }

        start = std::chrono::steady_clock::now();

        if (!device_.retrieve(color)) {
            return false;
        }

        // The timestamp of the driver, in milliseconds, or the time of reading if there is none
        const double milliseconds = device_.get(cv::CAP_PROP_POS_MSEC);

        time_ = milliseconds > 0.0
                  ? 1e-3 * milliseconds
                  : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

    } else {
        if (nextFile_ == files_.size()) {
            return false;
        }

        time_ = times_[nextFile_];
        color = cv::imread(files_[nextFile_++], cv::IMREAD_COLOR);

        if (color.empty()) {
            return false;
        }
    }

    cv::resize(color, color, frameSize_);
    cv::cvtColor(color, frame, cv::COLOR_BGR2GRAY);

    decodeTime_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++numRead_;

    return true;
}
