#ifndef MONOSLAM_STATE_STREAM_H
#define MONOSLAM_STATE_STREAM_H

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "display.h"
#include "ekf.h"
#include "feature.h"

namespace monoslam {

/*
 * Writes the state of every frame as text, for tools/viewer.py to show the scene as the filter sees it. The stream
 * starts with the camera,
 *
 *     camera <width> <height> <fx> <fy> <cx> <cy>
 *
 * the pinhole that the frames of the stream are undistorted to, and each frame is
 *
 *     frame <index> <rx> <ry> <rz> <ox> <oy> <oz> <used> <position std> <time>
 *     image <JPEG, base64>
 *     patch <id> <size> <pixels, base64>
 *     ...
 *     feature <id> <status> <x> <y> <z> <near x> <near y> <near z> <far x> <far y> <far z> <corners>
 *     ...
 *     match <prediction x y> <seed x y> <match x y> <ellipse xx xy yy>
 *     ...
 *     end
 *
 * The frame line has the camera position, its orientation as a rotation vector (camera to world), the number
 * of features the filter used, the largest standard deviation of the camera position, and the time the frame was
 * captured, in seconds, as FrameSource::time. The image line holds the frame, undistorted. The patch of every new
 * feature is written once, in the frame the feature is added in: its size and its pixels, row by row. Every feature has
 * a line with its id, which does not change when other features are removed, and its status (0 out of view, 1 in view,
 * 2 used by the update). Then come its position, the nearest and farthest points of its ray within two standard
 * deviations of its inverse depth, and the four corners of its patch, as Feature::patchCorners, in twelve coordinates.
 * Every match the filter used has a line with what the monoslam window draws, in the pixels of the undistorted frame:
 * the predicted measurement, the seed, the match, and the 99% ellipse of the innovation covariance. The ellipse
 * is given as its matrix E, the ellipse being (p - prediction)^T E^-1 (p - prediction) = 1.
 */
class StateStream
{
  public:
    /*
     * camera    Camera model
     */
    explicit StateStream(const CameraModel& camera);

    /*
     * Opens the file or pipe to write to, and writes the camera. Returns false if it cannot be opened.
     *
     * path    File, a named pipe, or the /dev/fd path of a process substitution
     */
    bool open(const std::string& path);

    bool isOpen() const { return file_.is_open(); }

    /*
     * Writes the state of a frame, and flushes it so that the viewer gets it right away.
     *
     * frame       Frame index
     * time        Time the frame was captured, in seconds
     * image       Frame
     * ekf         Filter
     * features    All the features
     * inView      Features in view
     * used        Features whose matches the filter used
     * matches     Matches the filter used, as drawn
     */
    void write(std::size_t frame,
               double time,
               const cv::Mat1b& image,
               const EKF& ekf,
               const std::vector<Feature>& features,
               const std::vector<std::size_t>& inView,
               const std::vector<std::size_t>& used,
               const std::vector<DrawnMatch>& matches);

  private:
    CameraModel camera_;
    cv::Mat mapX_, mapY_;  // Undistortion maps of the frames, to the pinhole of the camera matrix

    std::ofstream file_;
    std::size_t nextPatch_ = 0;  // Features with this id or larger have not had their patch written
};

}

#endif
