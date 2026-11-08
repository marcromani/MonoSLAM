#include "feature.h"

#include <cmath>

#include "relocalization.h"

monoslam::Feature::Feature(const cv::Mat1b& image,
                           const cv::Point2d& pixel,
                           const cv::Vec3d& patchNormal,
                           const CameraPose& firstPose,
                           std::size_t uniqueId)
  : frame(image)
  , position(pixel)
  , roi(static_cast<int>(std::lround(pixel.x)) - patchSize / 2,
        static_cast<int>(std::lround(pixel.y)) - patchSize / 2,
        patchSize,
        patchSize)
  , descriptors(describe(image, pixel))
  , normal(patchNormal)
  , pose(firstPose)
  , id(uniqueId)
{
}
