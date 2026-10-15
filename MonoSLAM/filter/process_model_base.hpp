#ifndef MONOSLAM_PROCESS_MODEL_BASE_H
#define MONOSLAM_PROCESS_MODEL_BASE_H

#include <opencv2/core.hpp>

namespace monoslam::filter {

class ProcessModelBase
{
  public:
    virtual ~ProcessModelBase() = default;

    /*
     * Returns f(x, 0), the prediction of the state transition function when evaluated at
     * the prior state x, under the assumption of zero noise w. Also, fills F = df/dx and
     * W = df/dw at the same point (x, 0).
     *
     * x     Prior state
     * dt    Elapsed time since the last iteration
     * F     Matrix of the linearized state transition function w.r.t. x
     * W     Matrix of the linearized state transition function w.r.t. w
     */
    virtual cv::Mat1d eval(const cv::Mat1d& x, double dt, cv::Mat1d& F, cv::Mat1d& W) const = 0;

    /*
     * Returns Q, the state transition noise covariance.
     *
     * dt    Elapsed time since the last iteration
     */
    virtual cv::Mat1d noiseCov(double dt) const = 0;
};

}

#endif
