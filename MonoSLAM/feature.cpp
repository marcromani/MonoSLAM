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

std::vector<cv::Point2f>
monoslam::Feature::patchPixels() const
{
    const cv::Point2f topLeft(roi.tl());
    const cv::Point2f bottomRight(roi.br());

    return { topLeft, cv::Point2f(bottomRight.x, topLeft.y), bottomRight, cv::Point2f(topLeft.x, bottomRight.y) };
}

std::vector<cv::Vec3d>
monoslam::Feature::patchCorners(const CameraModel& camera, const cv::Vec3d& point) const
{
    const double m = normal.dot(point - pose.r());

    std::vector<cv::Vec3d> corners;

    for (const cv::Point2f& pixel : patchPixels()) {
        const cv::Vec3d direction = pose.R() * camera.unproject(pixel);
        corners.push_back(pose.r() + direction * (m / normal.dot(direction)));
    }

    return corners;
}
