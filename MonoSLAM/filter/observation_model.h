#ifndef MONOSLAM_OBSERVATION_MODEL_H
#define MONOSLAM_OBSERVATION_MODEL_H

#include <cstddef>
#include <functional>
#include <vector>

#include <opencv2/core.hpp>

namespace monoslam::filter {

struct ObservationPrediction
{
    cv::Mat1d h;  // Predicted measurements
                  // Dim(h) = r x 1

    cv::Mat1d H;  // Jacobian of the measurement function w.r.t. x (H = dh/dx)
                  // Dim(H) = r x s

    cv::Mat1d V;  // Jacobian of the measurement function w.r.t. v (V = dh/dv)
                  // If the matrix is empty, it is assumed the noise is additive (i.e. V = I)
                  // Dim(V) = r x p

    std::vector<size_t> active;  // Sorted indices of the "active" entries of the state x regarding the measurement h
                                 // The indices identify which entry of the state x each column of H corresponds to
                                 // If the vector is empty, it is assumed H is full-width (i.e. s = n)

    cv::Mat1d R;  // Measurement noise covariance
                  // Dim(R) = p x p
};

class ObservationModel
{
  public:
    virtual ~ObservationModel() = default;

    /*
     * Decides which measurements exist for the state x and returns the corresponding data.
     * See ObservationPrediction.
     *
     * x    State, of size n x 1
     */
    virtual ObservationPrediction eval(const cv::Mat1d& x) const = 0;
};

struct Measurement
{
    cv::Mat1d z;  // Actual measurements
                  // Dim(z) = d x 1 (d <= r)

    std::vector<size_t> measured;  // Sorted indices of the d rows of h that were measured
                                   // It is assumed measured.size() = z.rows, so an empty vector means no measurement
};

using GatedMeasurementFunction =
  std::function<Measurement(const ObservationPrediction& prediction, const cv::Mat1d& S)>;

}

#endif
