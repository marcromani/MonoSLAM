#ifndef MONOSLAM_OBSERVATION_MODEL_BASE_H
#define MONOSLAM_OBSERVATION_MODEL_BASE_H

#include <opencv2/core.hpp>

namespace monoslam::filter {

class ObservationModelBase
{
  public:
    virtual ~ObservationModelBase() = default;

    /*
     * Returns h(x, 0), the measurement function evaluated at the predicted state x,
     * under the assumption of zero noise v. Also, fills H = dh/dx and V = dh/dv at
     * the same point (x, 0).
     *
     * x    Predicted state
     * H    Matrix of the linearized measurement function w.r.t. x
     * V    Matrix of the linearized measurement function w.t.t. v
     */
    virtual cv::Mat1d eval(const cv::Mat1d& x, cv::Mat1d& H, cv::Mat1d& V) const = 0;

    /*
     * Returns R, the measurement noise covariance.
     */
    virtual cv::Mat1d noiseCov() const = 0;
};

}

#endif
