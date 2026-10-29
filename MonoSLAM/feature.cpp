#include "feature.h"

#include "util.h"

monoslam::Feature::Feature(const cv::Mat1b& frame_,
                           const cv::Point2d& position_,
                           int patchSize_,
                           const cv::Vec3d& normal_,
                           const cv::Matx33d& R_,
                           const cv::Vec3d& t_)
  : frame(frame_.clone())
  , position(position_)
  , normal(normal_)
  , R(R_)
  , t(t_)
  , matchingFailures(0)
  , matchingAttempts(1)
  , initialized(true)
{
    roi = monoslam::util::buildSquare(position_, patchSize_);
}

monoslam::Feature::Feature(const cv::Mat1b& frame_,
                           const cv::Point2d& position_,
                           int patchSize_,
                           const cv::Matx33d& R_,
                           const cv::Vec3d& t_,
                           const cv::Vec3d& initDirection_,
                           const cv::Point2d& initDepthRange_)

  : frame(frame_.clone())
  , position(position_)
  , R(R_)
  , t(t_)
  , initDirection(initDirection_)
  , initialized(false)
{
    roi = monoslam::util::buildSquare(position_, patchSize_);

    const double step = (initDepthRange_.y - initDepthRange_.x) / 99;

    for (int i = 0; i < 100; ++i) {
        initDepths[i] = initDepthRange_.x + step * i;
    }

    initDepthProbs = cv::Vec<double, 100>::all(0.01);
}
