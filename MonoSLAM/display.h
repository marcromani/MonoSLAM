#ifndef MONOSLAM_DISPLAY_H
#define MONOSLAM_DISPLAY_H

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "ekf.h"

namespace monoslam {

/*
 * A match the filter used, as drawn: the predicted measurement with the 99% ellipse of its innovation covariance,
 * the seed, and the match.
 */
struct DrawnMatch
{
    cv::RotatedRect ellipse;
    cv::Point2d prediction;
    cv::Point2d seed;
    cv::Point2d match;
};

/*
 * What a frame shows: the image and the matches drawn on it.
 */
struct Snapshot
{
    cv::Mat1b frame;  // Shared with tracking: frames are never written to
    std::vector<DrawnMatch> matches;
    double scale = 1.0;  // From the pixels of the matches to those of the frame, if it is enlarged
};

/*
 * Returns the matches the filter used, to draw.
 *
 * o          Features in view, and their predicted measurements
 * seeds      Seeds of all the features
 * matches    Pixel of each feature of o, if it was found
 * used       Features the filter used
 */
std::vector<DrawnMatch>
usedMatches(const Observation& o,
            const std::vector<cv::Point2f>& seeds,
            const std::vector<std::optional<cv::Point2d>>& matches,
            const std::vector<std::size_t>& used);

/*
 * Draws a snapshot: the predicted measurement (cyan) with its ellipse, the seed (magenta), the match (amber),
 * and the innovation from the prediction to the match (white).
 */
cv::Mat3b
draw(const Snapshot& snapshot);

/*
 * Hands the frames from the tracking thread to the one that shows them, so that drawing and showing never delay
 * tracking. It holds only the last frame: a frame that the display had no time to show is replaced by the next one,
 * and tracking never waits for it.
 */
class Display
{
  public:
    /*
     * Replaces the frame to show. Never blocks for long: it only moves the snapshot.
     */
    void post(Snapshot snapshot);

    /*
     * Returns the frame to show, if a new one was posted since the last call.
     */
    std::optional<Snapshot> take();

    /*
     * Asks both threads to stop: the display when tracking ends, tracking when the user quits.
     */
    void stop() { stopped_ = true; }

    bool stopped() const { return stopped_; }

    /*
     * Returns how many frames have been taken to be shown.
     */
    std::size_t numShown() const { return numShown_; }

  private:
    std::mutex mutex_;  // Of latest_
    std::optional<Snapshot> latest_;
    std::atomic<bool> stopped_ = false;
    std::atomic<std::size_t> numShown_ = 0;
};

}

#endif
