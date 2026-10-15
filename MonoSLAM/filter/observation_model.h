#ifndef MONOSLAM_OBSERVATION_MODEL_H
#define MONOSLAM_OBSERVATION_MODEL_H

#include <opencv2/core.hpp>

namespace monoslam::filter {

struct Prediction
{
    cv::Mat1d h;  // Predicted measurements, of size r x 1
    cv::Mat1d H;  // Matrix of the linearized measurement function w.r.t. x, of size r x s

    std::vector<int> cols;  // Indices of the "measurable" entries of the state x, which correspond to the columns of H
                            // If cols is empty it is assumed H is full-width

    cv::Mat1d V;  // Matrix of the linearized measurement function w.r.t. v, of size r x b
                  // If V is empty it is assumed the noise is additive (i.e. V = I)

    cv::Mat1d R;  // Measurement noise covariance, of size b x b
};

class ObservationModel
{
  public:
    virtual ~ObservationModel() = default;

    /*
     * Decides which measurements exist for the state x and returns the corresponding data.
     * See filter::Prediction.
     *
     * x    State, of size n x 1
     */
    virtual Prediction eval(const cv::Mat1d& x) const = 0;
};

}

#endif
