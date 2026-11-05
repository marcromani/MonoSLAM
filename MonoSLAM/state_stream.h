#ifndef MONOSLAM_STATE_STREAM_H
#define MONOSLAM_STATE_STREAM_H

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ekf.h"
#include "feature.h"

namespace monoslam {

/*
 * Writes the state of every frame as text, for tools/viewer.py to show the scene as the filter
 * sees it. Each frame is
 *
 *     frame <index> <rx> <ry> <rz> <ox> <oy> <oz> <used> <position std>
 *     feature <id> <status> <x> <y> <z> <near x> <near y> <near z> <far x> <far y> <far z>
 *     ...
 *     end
 *
 * with the camera position, its orientation as a rotation vector (camera to world), the number of
 * features the filter used, and the largest standard deviation of the camera position. For every
 * feature, its id, which does not change when other features are removed, its status (0 out of
 * view, 1 in view, 2 used by the update), its position, and the nearest and farthest points of its
 * ray within two standard deviations of its inverse depth.
 */
class StateStream
{
  public:
    /*
     * Opens the file or pipe to write to. Returns false if it cannot be opened.
     *
     * path    File, a named pipe, or the /dev/fd path of a process substitution
     */
    bool open(const std::string& path);

    bool isOpen() const { return file_.is_open(); }

    /*
     * Writes the state of a frame, and flushes it so that the viewer gets it right away.
     *
     * frame       Frame index
     * ekf         Filter
     * features    All the features
     * inView      Features in view
     * used        Features whose matches the filter used
     */
    void write(std::size_t frame,
               const EKF& ekf,
               const std::vector<Feature>& features,
               const std::vector<std::size_t>& inView,
               const std::vector<std::size_t>& used);

  private:
    std::ofstream file_;
};

}

#endif
