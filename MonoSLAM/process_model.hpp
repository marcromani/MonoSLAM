#ifndef MONOSLAM_PROCESS_MODEL_H
#define MONOSLAM_PROCESS_MODEL_H

#include <opencv2/core.hpp>

#include "filter/process_model_base.hpp"

namespace monoslam {

class ProcessModel : public filter::ProcessModelBase
{
  public:
    ProcessModel(double linAccStd, double angAccStd);

    cv::Mat1d eval(const cv::Mat1d& x, double dt, cv::Mat1d& F, cv::Mat1d& W) const override;
    cv::Mat1d noiseCov(double dt) const override;

  private:
    cv::Mat1d f(const cv::Mat1d& x, double dt) const;
    cv::Mat1d computeF(const cv::Mat1d& x, double dt) const;
    cv::Mat1d computeW(double dt, const cv::Mat1d& F) const;

    cv::Matx66d acc_;
};

}

#endif
