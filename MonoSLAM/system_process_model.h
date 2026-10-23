#ifndef MONOSLAM_SYSTEM_PROCESS_MODEL_H
#define MONOSLAM_SYSTEM_PROCESS_MODEL_H

#include "filter/process_model.h"

namespace monoslam {

class SystemProcessModel : public filter::ProcessModel
{
  public:
    SystemProcessModel(double linearAccStd, double angularAccStd);

    filter::ProcessPrediction eval(const cv::Mat1d& x, double dt) const override;
    cv::Mat1d noiseCov(double dt) const override;
    std::vector<size_t> activeEntries() const override;

  private:
    cv::Mat1d f(const cv::Mat1d& x, double dt) const;
    cv::Mat1d computeF(const cv::Mat1d& x, double dt) const;
    cv::Mat1d computeW(double dt, const cv::Mat1d& F) const;

    cv::Matx66d acc_;
};

}

#endif
