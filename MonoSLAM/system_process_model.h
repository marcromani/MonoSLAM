#ifndef MONOSLAM_SYSTEM_PROCESS_MODEL_H
#define MONOSLAM_SYSTEM_PROCESS_MODEL_H

#include <opencv2/core.hpp>

#include "filter/process_model.h"

namespace monoslam {

class SystemProcessModel : public filter::ProcessModel
{
  public:
    SystemProcessModel(double linearAccStd, double angularAccStd);

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
