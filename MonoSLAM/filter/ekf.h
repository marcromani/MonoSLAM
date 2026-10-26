#ifndef MONOSLAM_EKF_H
#define MONOSLAM_EKF_H

#include <memory>

#include <opencv2/core.hpp>

#include "observation_model.h"
#include "process_model.h"

namespace monoslam::filter {

/*
 * Implements the standard Extended Kalman Filter. It does not support manifold operations.
 * That is, both the system state space and the system measurement space are assumed to be
 * Euclidean vector spaces.
 */
class EKF
{
  public:
    EKF(const std::shared_ptr<ProcessModel>&, const std::shared_ptr<ObservationModel>&);

    void init(const cv::Mat1d& x, const cv::Mat1d& P);

    void predict(double dt);
    void update(const MeasurementFunction& measure);

  private:
    std::shared_ptr<ProcessModel> processModel_;
    std::shared_ptr<ObservationModel> observationModel_;

    cv::Mat1d x_;  // System state
    cv::Mat1d P_;  // System covariance
};

}

#endif
