#ifndef MONOSLAM_EKF_H
#define MONOSLAM_EKF_H

#include <memory>

#include <opencv2/core.hpp>

#include "observation_model.h"
#include "process_model.h"

namespace monoslam::filter {

class EKF
{
  public:
    EKF(const std::shared_ptr<ProcessModel>&, const std::shared_ptr<ObservationModel>&);

    void predict(double dt);

  private:
    std::shared_ptr<ProcessModel> processModel_;
    std::shared_ptr<ObservationModel> observationModel_;

    cv::Mat1d x_;  // System state
    cv::Mat1d P_;  // System covariance
};

}

#endif
