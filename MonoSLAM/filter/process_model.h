#ifndef MONOSLAM_PROCESS_MODEL_H
#define MONOSLAM_PROCESS_MODEL_H

#include <opencv2/core.hpp>

namespace monoslam::filter {

struct ProcessPrediction
{
    cv::Mat1d x;  // Predicted state, according to the state transition x_{i+1} = f(x_i, w_i)
                  // Dim(x) = n x 1

    cv::Mat1d F;  // Jacobian of the state transition function w.r.t. x (F = df/dx)
                  // Dim(F) = k x k

    cv::Mat1d W;  // Jacobian of the state transition function w.r.t. w (W = df/dw)
                  // If the matrix is empty it is assumed the noise is additive (i.e. W = I)
                  // Dim(W) = k x b

    cv::Mat1d Q;  // State transition noise covariance
                  // Dim(Q) = b x b

    std::vector<int> active;  // Indices of the "active" entries of the state x, which correspond to the rows of F and W
                              // If the vector is empty it is assumed F and W are full-height (i.e. k = n)
                              // Len(active) = k
};

class ProcessModel
{
  public:
    virtual ~ProcessModel() = default;

    /*
     * Returns the predicted state.
     * See filter::ProcessPrediction.
     *
     * x     State, of size n x 1
     * dt    Elapsed time since the last iteration
     */
    virtual ProcessPrediction eval(const cv::Mat1d& x, double dt) const = 0;

    /*
     * Returns Q, the state transition noise covariance, of size m x m.
     *
     * dt    Elapsed time since the last iteration
     */
    virtual cv::Mat1d noiseCov(double dt) const = 0;

    /*
     * Returns k, the number of entries of x on which the model is different from the identity.
     * Note that it is implicitly assumed that these "active" entries are grouped together at
     * the front of the state vector x. TODO: This definition is not quiet right
     */
    virtual int activeDim(const cv::Mat1d& x) const { return x.rows; };
};

}

#endif
