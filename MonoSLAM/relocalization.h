#ifndef MONOSLAM_RELOCALIZATION_H
#define MONOSLAM_RELOCALIZATION_H

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "ekf.h"
#include "feature.h"

/*
 * Finding the features of the map anywhere in a frame, by their appearance alone, without the
 * state of the camera (Williams, Klein and Reid, 2011). It is what recovers the camera when
 * tracking is lost, and what finds the features the camera comes back to after a long excursion,
 * when the state has drifted too far for their seeds.
 *
 * The appearance of a feature is an ORB descriptor (Rublee et al., 2011): 256 binary comparisons of
 * pixel pairs of the patch around it, rotated with the patch, so that it does not change when the
 * image rotates, and little with moderate changes of scale and viewpoint.
 */
namespace monoslam {

/*
 * Returns the descriptors of some pixels of a frame, one per row, and removes the pixels too close
 * to the border to have one.
 *
 * frame        Grayscale frame
 * positions    Pixels
 */
cv::Mat1b
describe(const cv::Mat1b& frame, std::vector<cv::Point2f>& positions);

/*
 * Returns the descriptor of a pixel, or an empty matrix if it is too close to the border.
 */
cv::Mat1b
describe(const cv::Mat1b& frame, const cv::Point2d& position);

/*
 * Features of the map found in a frame by matchGlobally, and the pose that explains them.
 */
struct GlobalMatch
{
    cv::Vec3d r;    // Camera position
    cv::Matx33d R;  // Camera orientation, from camera to world

    std::vector<std::size_t> features;  // Features found
    std::vector<cv::Point2d> pixels;    // Where they were found
};

/*
 * Matches the corners of a frame to the well located features by their descriptors, and fits a
 * pose to the matches with P3P and RANSAC, taking the positions of the features as exact. The pose
 * is accepted if the patches of the features are also found where it projects them. Returns
 * nothing if too few features agree on a pose.
 *
 * frame       Current frame
 * features    All the features
 * camera      Camera model
 * ekf         Filter
 */
std::optional<GlobalMatch>
matchGlobally(const cv::Mat1b& frame, const std::vector<Feature>& features, const CameraModel& camera, const EKF& ekf);

}

#endif
