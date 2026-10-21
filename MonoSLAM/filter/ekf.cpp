#include "ekf.h"

#include "../util.h"
#include "observation_model.h"
#include "process_model.h"

void
monoslam::filter::EKF::predict(double dt)
{
    const ProcessPrediction p = processModel_->eval(x_, dt);

    x_ = p.x;

    cv::Mat1d N;  // TODO: Can it be p.active not empty (k < n) and p.W empty (k = b, w additive)?

    if (p.W.empty()) {
        N = p.Q;

    } else {
        N = p.W * p.Q * p.W.t();
    }

    if (p.active.empty() || p.active.size() == p.x.rows) {
        P_ = p.F * P_ * p.F.t() + N;

    } else {
        cv::Mat1d Paa = monoslam::util::selectRowsCols(P_, p.active, p.active);
        Paa = p.F * Paa * p.F.t() + N;

        std::vector<int> inactive = monoslam::util::complement(p.active, P_.rows);

        cv::Mat1d Pai = monoslam::util::selectRowsCols(P_, p.active, inactive);
        Pai = p.F * Pai;

        for (int i = 0; i < p.active.size(); ++i) {
            for (int j = 0; j < p.active.size(); ++j) {
                P_(p.active[i], p.active[j]) = Paa(i, j);
            }

            for (int j = 0; j < inactive.size(); ++j) {
                P_(p.active[i], inactive[j]) = Pai(i, j);
                P_(inactive[j], p.active[i]) = Pai(i, j);
            }
        }
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
monoslam::filter::EKF::update(const GatedMeasurementFunction& measure)
{
    const ObservationPrediction p = observationModel_->eval(x_);

    if (p.h.empty()) {
        return;
    }

    cv::Mat1d S;

    if (p.V.empty()) {
        S = p.H * P_ * p.H.t() + p.R;

    } else {
        S = p.H * P_ * p.H.t() + p.V * p.R * p.V.t();
    }

    const Measurement measurement = measure(p, S);

    cv::Mat1d residuals = measurement.z - util::selectRows(p.h, measurement.rows);

    cv::Mat1d H = util::selectRows(p.H, measurement.rows);

    cv::Mat1d P = util::selectRowsCols(P, p.cols);

    H * P * H.t();
}
