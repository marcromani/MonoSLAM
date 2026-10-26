#ifndef MONOSLAM_PROCESS_MODEL_H
#define MONOSLAM_PROCESS_MODEL_H

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace monoslam::filter {

struct ProcessPrediction
{
    cv::Mat1d x;  // Predicted state, according to the state transition x_{i+1} = f(x_i, w_i)
                  // Dim(x) = n x 1

    cv::Mat1d F;  // Jacobian of the state transition function w.r.t. x (F = df/dx)
                  // Dim(F) = k x k

    cv::Mat1d W;  // Jacobian of the state transition function w.r.t. w (W = df/dw)
                  // If the matrix is empty, it is assumed the noise is additive (i.e. W = I)
                  // Dim(W) = k x b
};

class ProcessModel
{
  public:
    virtual ~ProcessModel() = default;

    /*
     * Predicts the next state and returns the corresponding data.
     * See ProcessPrediction.
     *
     * x     State, of size n x 1
     * dt    Elapsed time since the last iteration
     */
    virtual ProcessPrediction eval(const cv::Mat1d& x, double dt) const = 0;

    /*
     * Returns Q, the state transition noise covariance, of size b x b. Note that if eval returns
     * an empty W (additive noise), b = k and dim(Q) = k × k, with k = activeEntries().size().
     *
     * dt    Elapsed time since the last iteration
     */
    virtual cv::Mat1d noiseCov(double dt) const = 0;

    /*
     * Returns the k sorted indices of the "active" entries of the state x (i.e. the entries that
     * have an actual effect as an input to the state transition function). The indices identify
     * which entry of the state x each column (row) of F corresponds to. For instance, if the
     * index vector is (2, 5, 6, 9), F has 4 columns (rows): the first one corresponds to the second
     * entry of x, the second one corresponds to the fifth entry of x, and so on. If the vector is empty,
     * it is assumed F is full-width (full-height), that is, k = n.
     */
    virtual std::vector<size_t> activeEntries() const { return {}; }

    virtual void enforceStateConstraints(cv::Mat1d& x, cv::Mat1d& P) const {}
};

}

#endif
