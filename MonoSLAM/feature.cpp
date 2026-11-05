#include "feature.h"

#include <cmath>

#include "relocalization.h"

monoslam::Feature::Feature(const cv::Mat1b& image,
                           const cv::Point2d& pixel,
                           int patchSize,
                           const cv::Vec3d& patchNormal,
                           const CameraPose& firstPose)
  : frame(image.clone())
  , position(pixel)
  , roi(static_cast<int>(std::lround(pixel.x)) - patchSize / 2,
        static_cast<int>(std::lround(pixel.y)) - patchSize / 2,
        patchSize,
        patchSize)
  , normal(patchNormal)
  , pose(firstPose)
{
    descriptors = monoslam::describe(frame, position);
}
