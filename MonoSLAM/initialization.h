#ifndef MONOSLAM_INITIALIZATION_H
#define MONOSLAM_INITIALIZATION_H

#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "feature.h"

namespace monoslam {

/*
 * Initializes the state with the calibration pattern: the camera pose comes from the whole pattern, and five
 * of its corners become the first features, at their known positions. This is where the absolute scale of the map comes
 * from. The pattern is not used afterwards. Returns false if the pattern is not found, its pose is not accurate enough,
 * or a corner is too close to the border of the frame to take its patch.
 *
 * frame          Current frame
 * patternSize    Number of inner corners of the pattern, per row and per column
 * squareSize     Side of a square of the pattern, in world units
 * camera         Camera model
 * x              Initial state
 * orientation    Initial orientation, from camera to world
 * P              Initial covariance
 * features       Initial features
 */
bool
initializeFromPattern(const cv::Mat1b& frame,
                      const cv::Size& patternSize,
                      double squareSize,
                      const CameraModel& camera,
                      cv::Mat1d& x,
                      cv::Matx33d& orientation,
                      cv::Mat1d& P,
                      std::vector<Feature>& features);

}

#endif
