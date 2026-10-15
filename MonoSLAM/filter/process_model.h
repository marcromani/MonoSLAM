#ifndef MONOSLAM_PROCESS_MODEL_H
#define MONOSLAM_PROCESS_MODEL_H

#include <opencv2/core.hpp>

namespace monoslam::filter {

class ProcessModel
{
  public:
    virtual ~ProcessModel() = default;

    /*
     * Returns xPred = f(x, w = 0), the prediction of the state transition function when evaluated
     * at the previous state x, under the assumption of zero noise w. Also, it fills F = df/dx and
     * W = df/dw, evaluated at the same point. Note that xPred is k x 1, F is k x k, and W is k x m,
     * where k = activeDim(x) and m is the dimension of the error w. If W is empty it is assumed
     * the noise is additive (i.e. W = I).
     *
     * x     State, of size n x 1
     * dt    Elapsed time since the last iteration
     * F     Matrix of the linearized state transition function w.r.t. x
     * W     Matrix of the linearized state transition function w.r.t. w
     */
    virtual cv::Mat1d eval(const cv::Mat1d& x, double dt, cv::Mat1d& F, cv::Mat1d& W) const = 0;

    /*
     * Returns Q, the state transition noise covariance, of size m x m.
     *
     * dt    Elapsed time since the last iteration
     */
    virtual cv::Mat1d noiseCov(double dt) const = 0;

    /*
     * Returns k, the number of entries of x on which the model is different from the identity.
     * Note that it is implicitly assumed that these "active" entries are grouped together at
     * the front of the state vector x.
     */
    virtual int activeDim(const cv::Mat1d& x) { return x.rows; };
};

}

#endif
