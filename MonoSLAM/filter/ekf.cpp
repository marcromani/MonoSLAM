#include "ekf.h"

#include <algorithm>
#include <cassert>

#include <opencv2/core.hpp>

#include "observation_model.h"
#include "process_model.h"
#include "util.h"

monoslam::filter::EKF::EKF(const std::shared_ptr<ProcessModel>& processModel,
                           const std::shared_ptr<ObservationModel>& observationModel)
  : processModel_(processModel)
  , observationModel_(observationModel)
{
}

void
monoslam::filter::EKF::init(const cv::Mat1d& x, const cv::Mat1d& P)
{
    x_ = x.clone();
    P_ = P.clone();
}

void
monoslam::filter::EKF::predict(double dt)
{
    const ProcessPrediction p = processModel_->eval(x_, dt);

    x_ = p.x;

    cv::Mat1d N;

    if (p.W.empty()) {
        N = processModel_->noiseCov(dt);

    } else {
        N = p.W * processModel_->noiseCov(dt) * p.W.t();
    }

    const std::vector<size_t> active = processModel_->activeEntries();

    if (active.empty() || active.size() == static_cast<size_t>(x_.rows)) {
        P_ = p.F * P_ * p.F.t() + N;

    } else {
        cv::Mat1d Paa = util::selectRowsCols(P_, active, active);
        Paa = p.F * Paa * p.F.t() + N;

        const std::vector<size_t> inactive = util::complement(active, static_cast<size_t>(x_.rows));

        cv::Mat1d Pai = util::selectRowsCols(P_, active, inactive);
        Pai = p.F * Pai;

        for (size_t i = 0; i < active.size(); ++i) {
            const int r = static_cast<int>(active[i]);

            for (size_t j = 0; j < active.size(); ++j) {
                const int c = static_cast<int>(active[j]);

                // Update the active part of the covariance
                P_(r, c) = Paa(static_cast<int>(i), static_cast<int>(j));
            }

            for (size_t j = 0; j < inactive.size(); ++j) {
                const int c = static_cast<int>(inactive[j]);

                // Update the cross terms with the inactive entries
                P_(r, c) = Pai(static_cast<int>(i), static_cast<int>(j));
                P_(c, r) = Pai(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    P_ = 0.5 * (P_ + P_.t());  // Symmetrize to avoid rounding errors over thousands of steps
}

void
monoslam::filter::EKF::update(const MeasurementFunction& measure)
{
    const ObservationPrediction p = observationModel_->eval(x_);

    if (p.h.empty()) {
        return;
    }

    const cv::Mat1d N = p.V.empty() ? p.R : p.V * p.R * p.V.t();

    const bool fullWidth = p.active.empty() || p.active.size() == static_cast<size_t>(x_.rows);

    cv::Mat1d Paa, Pcols;

    if (fullWidth) {
        Paa = P_;
        Pcols = P_;

    } else {
        Paa = util::selectRowsCols(P_, p.active, p.active);
        Pcols = util::selectCols(P_, p.active);
    }

    cv::Mat1d S = p.H * Paa * p.H.t() + N;

    const Measurement measurement = measure(x_, p, S);

    if (measurement.measured.empty()) {
        return;
    }

    assert(static_cast<int>(measurement.measured.size()) == measurement.z.rows);
    assert(std::is_sorted(measurement.measured.begin(), measurement.measured.end()));
    assert(measurement.measured.back() < static_cast<size_t>(p.h.rows));
    assert(std::adjacent_find(measurement.measured.begin(), measurement.measured.end(), [](size_t a, size_t b) {
               return a == b;
           }) == measurement.measured.end());

    cv::Mat1d h = p.h;
    cv::Mat1d H = p.H;

    if (measurement.measured.size() != static_cast<size_t>(p.h.rows)) {
        h = util::selectRows(h, measurement.measured);
        H = util::selectRows(H, measurement.measured);
        S = util::selectRowsCols(S, measurement.measured, measurement.measured);
    }

    const cv::Mat1d PHt = Pcols * H.t();

    cv::Mat1d Kt;

    if (!cv::solve(S, PHt.t(), Kt, cv::DECOMP_CHOLESKY)) {
        return;
    }

    x_ += Kt.t() * (measurement.z - h);
    P_ -= PHt * Kt;

    processModel_->enforceStateConstraints(x_, P_);

    P_ = 0.5 * (P_ + P_.t());  // Symmetrize to avoid rounding errors over thousands of steps
}
