#ifndef MONOSLAM_TRACKING_H
#define MONOSLAM_TRACKING_H

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "ekf.h"
#include "feature.h"

namespace monoslam {

/*
 * Follows the image motion of every feature from frame to frame, independently of the filter, so
 * that each feature can be searched in a small region around where it moved, its seed. The motion
 * model of the filter cannot predict a jerk of the hand, but the images can.
 */
class FlowTracker
{
  public:
    /*
     * frame        First frame
     * positions    Feature positions in it, in pixels
     */
    FlowTracker(const cv::Mat1b& frame, const std::vector<cv::Point2f>& positions);

    /*
     * Moves the seeds to a new frame and returns them. Each seed follows an affine motion fitted
     * to the generic corners around it, which unlike a global model does not assume anything about
     * the structure of the scene, only that close points in the image move alike. The flow of the
     * feature itself takes over only if it agrees with that motion, since the flow of a single
     * point is more precise, but it can also jump to a look-alike neighbor.
     *
     * frame    New frame
     */
    std::vector<cv::Point2f>& predict(const cv::Mat1b& frame);

    /*
     * Replaces the seeds with better positions. Those coming from the filter state make the
     * features agree with each other, so that a feature that slipped cannot stay lost.
     */
    void correct(const std::vector<cv::Point2f>& positions) { seeds_ = positions; }

    /*
     * Adds the seed of a new feature, in the last frame.
     */
    void add(const cv::Point2f& position) { seeds_.push_back(position); }

    /*
     * Removes the seeds of some features, given by their sorted indices.
     */
    void remove(const std::vector<std::size_t>& features);

  private:
    cv::Mat1b frame_;                 // Last frame
    std::vector<cv::Point2f> seeds_;  // Feature positions in the last frame
};

/*
 * Searches a feature around a pixel, and returns where it is found, if anywhere. The patch of the
 * feature is first warped to how it would look from the current pose, with the homography of the
 * plane of the patch between the two views. The search is a zero-mean normalized cross-correlation
 * over the warped patch, and the peak is refined to subpixel precision with a parabola. Returns
 * nothing if the best score is low, or if another peak scores almost as high.
 *
 * frame       Current frame
 * feature     Feature to look for
 * position    Feature position, in world coordinates
 * camera      Camera model
 * pose        Camera pose
 * center      Center of the search region, in pixels
 * radius      Radius of the search region, in pixels
 */
std::optional<cv::Point2d>
findPatch(const cv::Mat1b& frame,
          const Feature& feature,
          const cv::Vec3d& position,
          const CameraModel& camera,
          const CameraPose& pose,
          const cv::Point2d& center,
          int radius);

/*
 * Searches the features in view in a frame, each in a small region around its seed, or around its
 * predicted measurement if its seed is not in view. Telling the right matches from the wrong ones
 * is left to the filter. Returns the pixel of each feature of o, if it was found.
 *
 * frame       Current frame
 * features    All the features
 * camera      Camera model
 * ekf         Filter, after the prediction
 * o           Features in view, and their predicted measurements
 * seeds       Expected image position of all the features
 */
std::vector<std::optional<cv::Point2d>>
matchFeatures(const cv::Mat1b& frame,
              const std::vector<Feature>& features,
              const CameraModel& camera,
              const EKF& ekf,
              const Observation& o,
              const std::vector<cv::Point2f>& seeds);

}

#endif
