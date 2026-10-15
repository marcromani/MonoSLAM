#include "ekf.h"

#include "observation_model.h"

void
monoslam::filter::EKF::predict(double dt)
{
    const int n = x_.rows;
    const int k = processModel_->activeDim(x_);

    cv::Mat1d F, W;
    x_ = processModel_->eval(x_, dt, F, W);

    // Covariance of the active entries of x
    const cv::Mat1b Paa = P_(cv::Rect(0, 0, k, k));

    cv::Mat1d N;

    if (W.empty()) {
        N = processModel_->noiseCov(dt);

    } else {
        N = W * processModel_->noiseCov(dt) * W.t();
    }

    // Update the active part of the covariance
    cv::Mat1d(F * Paa * F.t() + N).copyTo(P_(cv::Rect(0, 0, k, k)));

    if (k < n) {
        cv::Mat1d Pai = F * P_(cv::Rect(k, 0, n - k, k));

        // Update the cross terms with the inactive entries
        Pai.copyTo(P_(cv::Rect(k, 0, n - k, k)));
        cv::Mat1d(Pai.t()).copyTo(P_(cv::Rect(0, k, k, n - k)));
    }
}

void
monoslam::filter::EKF::update()
{
    const Prediction p = observationModel_->eval(x_);

    if (p.h.empty()) {
        return;
    }

    cv::Mat1d S;

    if (p.V.empty()) {
        S = p.H * P_ * p.H.t() + p.R;

    } else {
        S = p.H * P_ * p.H.t() + p.V * p.R * p.V.t();
    }
}
