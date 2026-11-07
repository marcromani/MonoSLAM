#ifndef MONOSLAM_FRAME_SOURCE_H
#define MONOSLAM_FRAME_SOURCE_H

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace monoslam {

/*
 * Provides the frames, either from a camera or from a directory of images. Every frame is
 * resized to the size of the calibration and converted to grayscale.
 */
class FrameSource
{
  public:
    /*
     * Opens a camera device. Returns false if it cannot be opened.
     *
     * device       Camera device
     * frameSize    Size of the frames
     * fps          Frames per second to ask the camera for
     */
    bool openCamera(const std::string& device, const cv::Size& frameSize, double fps);

    /*
     * Opens a directory of images, which are read in alphabetical order. Their capture times are
     * read from the column capture_us of a file timestamps.csv in the directory, as record_sequence
     * writes it, if there is one, and are those of a constant frame rate otherwise. Returns false if
     * there are no images in it.
     *
     * directory    Directory of the images
     * frameSize    Size of the frames
     * fps          Frame rate to assume without capture times
     */
    bool openImages(const std::string& directory, const cv::Size& frameSize, double fps);

    /*
     * Reads the next frame. With a camera, it waits until there is one. With images, it returns
     * false once they have all been read.
     *
     * frame    Grayscale frame
     */
    bool read(cv::Mat1b& frame);

    /*
     * Returns the index of the last frame read, counting from 0.
     */
    std::size_t index() const { return numRead_ - 1; }

    /*
     * Returns the time at which the last frame read was captured, in seconds, from an arbitrary
     * origin. With a camera it is the timestamp of the driver, not the time at which the frame was
     * read, which can be much later: frames wait in a buffer, and are dropped while the previous
     * one is processed.
     */
    double time() const { return time_; }

    /*
     * Returns how long the last frame took to decode and convert, in seconds, without the wait for
     * the camera to deliver it.
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
