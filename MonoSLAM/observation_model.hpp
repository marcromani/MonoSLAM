#ifndef MONOSLAM_OBSERVATION_MODEL_H
#define MONOSLAM_OBSERVATION_MODEL_H

#include <opencv2/core.hpp>

#include "filter/observation_model_base.hpp"

namespace monoslam {

class ObservationModel : public filter::ObservationModelBase
{
  public:
    ObservationModel(const cv::Matx33d& K);

    cv::Mat1d eval(const cv::Mat1d& x, cv::Mat1d& H, cv::Mat1d& V) const override;
    cv::Mat1d noiseCov() const override;

  private:
    cv::Mat1d h(const cv::Mat1d& x) const;
    cv::Mat1d computeH(const cv::Mat1d& x) const;

    cv::Matx33d K_;
};

}
#endif
