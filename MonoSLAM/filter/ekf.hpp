#ifndef MONOSLAM_EKF_H
#define MONOSLAM_EKF_H

#include <memory>

#include <opencv2/core.hpp>

#include "observation_model_base.hpp"
#include "process_model_base.hpp"

namespace monoslam::filter {

class EKF
{
  public:
    EKF(const std::shared_ptr<ProcessModelBase>&, const std::shared_ptr<ObservationModelBase>&);

    void predict();
    void update();

  private:
    std::shared_ptr<ProcessModelBase> processModel_;
    std::shared_ptr<ObservationModelBase> observationModel_;

    cv::Mat1d x_;  // System state
    cv::Mat1d P_;  // System covariance
};

}

#endif
