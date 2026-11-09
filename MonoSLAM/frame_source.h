#ifndef MONOSLAM_FRAME_SOURCE_H
#define MONOSLAM_FRAME_SOURCE_H

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace monoslam {

constexpr double cameraFps = 120.0;      // Asked from the camera, which takes the nearest rate it has
constexpr double defaultExposure = 5.0;  // Milliseconds, short enough not to lower the frame rate

/*
 * Provides the frames, either from a camera or from a directory of images, in grayscale and at the size
 * of the calibration, as they come: they are never resized.
 */
class FrameSource
{
  public:
    /*
     * Opens a camera device, asking it for frames of frameSize at cameraFps, with a fixed exposure. With automatic
     * exposure the camera would lengthen it in dim light, and lower the frame rate to make room for it. Returns false,
     * saying why, if it cannot be opened or does not give frames of frameSize.
     *
     * device       Camera device
     * frameSize    Size of the frames returned
     * exposure     Exposure time, in milliseconds
     */
    bool openCamera(const std::string& device, const cv::Size& frameSize, double exposure);

    /*
     * Opens a directory of images, which are read in alphabetical order. Their capture times are read from the column
     * capture_us of a file timestamps.csv in the directory, as record_sequence writes it, if there is one,
     * and are those of a constant frame rate otherwise. Returns false if there are no images in it.
     *
     * directory    Directory of the images
     * frameSize    Size of the frames
     * fps          Frame rate to assume without capture times
     */
    bool openImages(const std::string& directory, const cv::Size& frameSize, double fps);

    /*
     * Reads the next frame. With a camera, it waits until there is one. With images, it returns false once they have
     * all been read, or at one of another size than frameSize.
     *
     * frame    Grayscale frame, a new image every time, which nothing writes to afterwards: it can be kept, and shared
     *          between threads, without copying it
     */
    bool read(cv::Mat1b& frame);

    /*
     * Returns the index of the last frame read, counting from 0.
     */
    std::size_t index() const { return numRead_ - 1; }

    /*
     * Returns the time at which the last frame read was captured, in seconds, from an arbitrary origin. With a camera
     * it is the timestamp of the driver, not the time at which the frame was read. That can be much later, since frames
     * wait in a buffer and are dropped while the previous one is processed.
     */
    double time() const { return time_; }

    /*
     * Returns how long the last frame took to decode and convert, in seconds, without the wait for the camera
     * to deliver it.
     */
    double decodeTime() const { return decodeTime_; }

  private:
    cv::VideoCapture device_;
    std::vector<std::string> files_;  // Images to read, empty with a camera
    std::size_t nextFile_ = 0;
    std::size_t numRead_ = 0;
    double time_ = 0.0;
    double decodeTime_ = 0.0;
    std::vector<double> times_;  // Of the images, in seconds
    cv::Size frameSize_;
};

}

#endif
