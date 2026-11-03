#ifndef MONOSLAM_SYSTEM_PROCESS_MODEL_H
#define MONOSLAM_SYSTEM_PROCESS_MODEL_H

#include "filter/process_model.h"

namespace monoslam {

inline constexpr int cameraStateSize = 7;
inline constexpr int firstFeatureState = cameraStateSize;

/*
 * The camera pose is modeled as a random walk, it is predicted to stay where it is,
 * with an uncertainty that grows with time. Features do not move.
 */
class SystemProcessModel : public filter::ProcessModel
{
  public:
    SystemProcessModel(double linearVelStd, double angularVelStd);

    filter::ProcessPrediction eval(const cv::Mat1d& x, double dt) const override;
    cv::Mat1d noiseCov(double dt) const override;
    std::vector<size_t> activeEntries() const override;
    void enforceStateConstraints(cv::Mat1d& x, cv::Mat1d& P) const override;

  private:
    cv::Mat1d computeW(const cv::Mat1d& x) const;

    cv::Matx66d vel_;
};

}

#endif
