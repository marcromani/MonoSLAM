#include "ekf.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

using Indices = std::vector<std::size_t>;

cv::Mat1d
selectRows(const cv::Mat1d& m, const Indices& rows)
{
    cv::Mat1d result(static_cast<int>(rows.size()), m.cols);

    for (std::size_t i = 0; i < rows.size(); ++i) {
        m.row(static_cast<int>(rows[i])).copyTo(result.row(static_cast<int>(i)));
    }

    return result;
}

cv::Mat1d
selectCols(const cv::Mat1d& m, const Indices& cols)
{
    cv::Mat1d result(m.rows, static_cast<int>(cols.size()));

    for (int r = 0; r < m.rows; ++r) {
        for (std::size_t j = 0; j < cols.size(); ++j) {
            result(r, static_cast<int>(j)) = m(r, static_cast<int>(cols[j]));
        }
    }

    return result;
}

cv::Mat1d
selectRowsCols(const cv::Mat1d& m, const Indices& rows, const Indices& cols)
{
    return selectCols(selectRows(m, rows), cols);
}

/*
 * Returns A H^T, skipping the zeros of H. A pixel depends on the camera and on its own feature only,
 * so most of H is zero, and with many features this saves most of the product.
 */
cv::Mat1d
multiplyTransposed(const cv::Mat1d& A, const cv::Mat1d& H)
{
    assert(A.cols == H.cols);

    std::vector<std::vector<int>> nonzeros(static_cast<std::size_t>(H.rows));  // Columns of each row of H

    for (int r = 0; r < H.rows; ++r) {
        for (int c = 0; c < H.cols; ++c) {
            if (H(r, c) != 0.0) {
                nonzeros[static_cast<std::size_t>(r)].push_back(c);
            }
        }
    }

    cv::Mat1d result(A.rows, H.rows);

    for (int i = 0; i < A.rows; ++i) {
        for (int r = 0; r < H.rows; ++r) {
            double sum = 0.0;

            for (int c : nonzeros[static_cast<std::size_t>(r)]) {
                sum += A(i, c) * H(r, c);
            }

            result(i, r) = sum;
        }
    }

    return result;
}

/*
 * Returns the state entries that H acts on: the camera, and the parameters of the features in view.
 */
Indices
columnsOf(const monoslam::Observation& o)
{
    Indices columns;

    for (int i = 0; i < monoslam::cameraStateSize; ++i) {
        columns.push_back(static_cast<std::size_t>(i));
    }

    for (std::size_t feature : o.features) {
        for (int k = 0; k < monoslam::featureSize; ++k) {
            columns.push_back(static_cast<std::size_t>(monoslam::featureIndex(feature) + k));
        }
    }

    return columns;
}

/*
 * Returns the rows of o.h of a feature, if o predicts it in view.
 */
std::optional<Indices>
rowsOf(const monoslam::Observation& o, std::size_t feature)
{
    const auto it = std::find(o.features.begin(), o.features.end(), feature);

    if (it == o.features.end()) {
        return std::nullopt;
    }

    const auto k = static_cast<std::size_t>(it - o.features.begin());

    return Indices{ 2 * k, 2 * k + 1 };
}

/*
 * Returns the factor a velocity decays by in a time step, and the time the velocity it had at the
 * start of the step would take to cover the same distance, tau (1 - exp(-dt / tau)).
 */
void
decay(double dt, double tau, double& factor, double& duration)
{
    factor = std::exp(-dt / tau);
    duration = tau * (1.0 - factor);
}

/*
 * Returns the right Jacobian of SO(3), J_r(v), with exp(v + d) = exp(v) exp(J_r(v) d) to first
 * order: J_r(v) = I - (1 - cos|v|) / |v|^2 [v]x + (|v| - sin|v|) / |v|^3 [v]x^2.
 */
cv::Matx33d
rightJacobian(const cv::Vec3d& v)
{
    const double angle = cv::norm(v);
    const cv::Matx33d V = monoslam::skew(v);

    if (angle < 1e-6) {
        return cv::Matx33d::eye() - 0.5 * V;
    }

    const double a = (1.0 - std::cos(angle)) / (angle * angle);
    const double b = (angle - std::sin(angle)) / (angle * angle * angle);

    return cv::Matx33d::eye() - a * V + b * V * V;
}

}

cv::Mat1d
monoslam::MotionModel::predict(const cv::Mat1d& x,
                               cv::Matx33d& R,
                               double dt,
                               cv::Matx<double, cameraStateSize, cameraStateSize>* F,
                               cv::Matx<double, cameraStateSize, 6>* W) const
{
    double linearFactor, linearDuration, angularFactor, angularDuration;
    decay(dt, linearDecayTime, linearFactor, linearDuration);
    decay(dt, angularDecayTime, angularFactor, angularDuration);

    // r_new = r_old + v_old h_v, R_new = R_old exp(w_old h_w), with h the durations of the decay.
    // w is in camera coordinates, so its rotation applies on the right
    const cv::Vec3d step = cv::Vec3d(x(9), x(10), x(11)) * angularDuration;
    const cv::Matx33d stepRotation = rotation(step);

    R = R * stepRotation;

    cv::Mat1d xNew = x.clone();

    xNew(cv::Rect(0, 0, 1, 3)) += xNew(cv::Rect(0, 6, 1, 3)) * linearDuration;
    xNew(cv::Rect(0, 6, 1, 3)) *= linearFactor;
    xNew(cv::Rect(0, 9, 1, 3)) *= angularFactor;

    if (F == nullptr && W == nullptr) {
        return xNew;
    }

    // The error after the step is e_new with R_new exp(e_new) = R_old exp(e) exp((w_old + dw) h),
    // which to first order is e_new = exp(-w_old h) e + J_r(w_old h) h dw
    const cv::Matx33d Jr = rightJacobian(step);

    if (F != nullptr) {
        *F = F->eye();

        for (int i = 0; i < 3; ++i) {
            // d r_new / d v_old, d v_new / d v_old and d w_new / d w_old
            (*F)(i, 6 + i) = linearDuration;
            (*F)(6 + i, 6 + i) = linearFactor;
            (*F)(9 + i, 9 + i) = angularFactor;

            // d e_new / d e and d e_new / d w_old
            for (int j = 0; j < 3; ++j) {
                (*F)(3 + i, 3 + j) = stepRotation(j, i);
                (*F)(3 + i, 9 + j) = Jr(i, j) * angularDuration;
            }
        }
    }

    if (W != nullptr) {
        *W = W->zeros();

        for (int i = 0; i < 3; ++i) {
            // The acceleration is constant during the step, so the position only gets half of the
            // velocity change on average: r_new = r_old + (v_old + big_v / 2) dt, and likewise the
            // rotation of the step is w_old h + big_w dt / 2
            (*W)(i, i) = 0.5 * dt;

            for (int j = 0; j < 3; ++j) {
                (*W)(3 + i, 3 + j) = Jr(i, j) * (0.5 * dt);
            }
        }

        // d v_new / d big_v and d w_new / d big_w
        for (int i = 0; i < 6; ++i) {
            (*W)(6 + i, i) = 1.0;
        }
    }

    return xNew;
}

cv::Matx66d
monoslam::MotionModel::noiseCovariance(double dt) const
{
    cv::Matx66d Q = cv::Matx66d::zeros();

    for (int i = 0; i < 3; ++i) {
        Q(i, i) = linearAccStd * linearAccStd;
        Q(3 + i, 3 + i) = angularAccStd * angularAccStd;
    }

    return Q * dt * dt;
}

monoslam::EKF::EKF(const MotionModel& motion,
                   const CameraModel& camera,
                   double pixelStd,
                   const cv::Mat1d& x,
                   const cv::Matx33d& R,
                   const cv::Mat1d& P)
  : motion_(motion)
  , camera_(camera)
  , pixelStd_(pixelStd)
  , x_(x.clone())
  , R_(R)
  , P_(P.clone())
{
}

void
monoslam::EKF::predict(double dt)
{
    constexpr int c = cameraStateSize;

    cv::Matx<double, c, c> F;
    cv::Matx<double, c, 6> W;
    x_ = motion_.predict(x_, R_, dt, &F, &W);

    // Only the camera moves: P becomes [F Pcc F^T + W Q W^T, F Pcf; Pfc F^T, Pff]
    const int n = x_.rows;

    cv::Matx<double, c, c> Pcc;
    P_(cv::Rect(0, 0, c, c)).copyTo(Pcc);
    Pcc = F * Pcc * F.t() + W * motion_.noiseCovariance(dt) * W.t();
    cv::Mat1d(Pcc).copyTo(P_(cv::Rect(0, 0, c, c)));

    if (n > c) {
        cv::Mat1d Pcf = P_(cv::Rect(c, 0, n - c, c)).clone();
        Pcf = cv::Mat1d(F) * Pcf;

        Pcf.copyTo(P_(cv::Rect(c, 0, n - c, c)));
        cv::Mat1d(Pcf.t()).copyTo(P_(cv::Rect(0, c, c, n - c)));
    }

    P_ = 0.5 * (P_ + P_.t());  // Symmetrize to avoid rounding errors over thousands of steps
}

monoslam::Observation
monoslam::EKF::predictMeasurements(const cv::Mat1d& x) const
{
    const CameraPose pose(cameraPosition(x), R_ * rotation(orientationError(x)));

    Observation o;

    std::vector<cv::Point2d> pixels;
    std::vector<cv::Matx<double, 2, cameraStateSize>> Jcamera;
    std::vector<cv::Matx<double, 2, featureSize>> Jfeature;

    for (std::size_t i = 0; i < numFeatures(x); ++i) {
        cv::Matx33d Jcr, Jce;
        cv::Matx<double, 3, featureSize> Jcy;

        const cv::Vec3d hc = inverse_depth::toCamera(pose, featureParameters(x, i), Jcr, Jce, Jcy);

        cv::Point2d pixel;
        cv::Matx23d Jpixel;

        if (camera_.projectCameraPoint(hc, pixel, Jpixel)) {
            // The pixel depends on the camera position and orientation, not on its velocities
            cv::Matx<double, 2, cameraStateSize> Jc = cv::Matx<double, 2, cameraStateSize>::zeros();
            const cv::Matx23d Jr = Jpixel * Jcr;
            const cv::Matx23d Je = Jpixel * Jce;

            for (int row = 0; row < 2; ++row) {
                for (int k = 0; k < 3; ++k) {
                    Jc(row, k) = Jr(row, k);
                    Jc(row, 3 + k) = Je(row, k);
                }
            }

            o.features.push_back(i);
            pixels.push_back(pixel);
            Jcamera.push_back(Jc);
            Jfeature.push_back(Jpixel * Jcy);
        }
    }

    const int m = static_cast<int>(o.features.size());

    o.h = cv::Mat1d(2 * m, 1);
    o.H = cv::Mat1d::zeros(2 * m, cameraStateSize + featureSize * m);

    for (int k = 0; k < m; ++k) {
        const auto j = static_cast<std::size_t>(k);

        o.h(2 * k) = pixels[j].x;
        o.h(2 * k + 1) = pixels[j].y;

        cv::Mat1d(Jcamera[j]).copyTo(o.H(cv::Rect(0, 2 * k, cameraStateSize, 2)));
        cv::Mat1d(Jfeature[j]).copyTo(o.H(cv::Rect(cameraStateSize + featureSize * k, 2 * k, featureSize, 2)));
    }

    return o;
}

monoslam::Observation
monoslam::EKF::observe() const
{
    Observation o = predictMeasurements(x_);

    if (o.features.empty()) {
        return o;
    }

    const Indices columns = columnsOf(o);
    const cv::Mat1d R(pixelStd_ * pixelStd_ * cv::Mat1d::eye(o.h.rows, o.h.rows));

    o.Pcols = selectCols(P_, columns);
    o.S = cv::Mat1d(o.H * selectRowsCols(P_, columns, columns) * o.H.t() + R);

    return o;
}

void
monoslam::EKF::correct(const Observation& o, const Indices& rows, const cv::Mat1d& z)
{
    const cv::Mat1d h = selectRows(o.h, rows);
    const cv::Mat1d H = selectRows(o.H, rows);
    const cv::Mat1d S = selectRowsCols(o.S, rows, rows);

    const cv::Mat1d PHt = multiplyTransposed(o.Pcols, H);

    // The transpose of the gain, K^T = S^-1 (P H^T)^T, since S is symmetric
    cv::Mat1d Kt;

    if (!cv::solve(S, PHt.t(), Kt, cv::DECOMP_CHOLESKY)) {
        return;
    }

    x_ += Kt.t() * (z - h);
    cv::gemm(PHt, Kt, -1.0, P_, 1.0, P_);  // P = P - P H^T K^T, in place since P is large

    // R absorbs the orientation error, which goes back to zero, and the uncertainty of the error
    // moves to the new reference: an error d about R exp(e) is one of G d, G = I - [e / 2]x, about R
    const cv::Vec3d e = orientationError(x_);
    const cv::Mat1d G(cv::Matx33d::eye() - 0.5 * skew(e));

    cv::Mat1d(G * P_.rowRange(3, 6)).copyTo(P_.rowRange(3, 6));
    cv::Mat1d(P_.colRange(3, 6) * G.t()).copyTo(P_.colRange(3, 6));

    R_ = R_ * rotation(e);
    x_(cv::Rect(0, 3, 1, 3)).setTo(0.0);
}

std::vector<std::size_t>
monoslam::EKF::update(const Observation& o,
                      const std::vector<std::optional<cv::Point2d>>& matches,
                      double maxDistance,
                      double maxNIS)
{
    assert(matches.size() == o.features.size());

    // Every match is a group of two rows of the observation
    std::vector<std::size_t> groups;  // Index in o.features of each match

    for (std::size_t k = 0; k < matches.size(); ++k) {
        if (matches[k].has_value()) {
            groups.push_back(k);
        }
    }

    if (groups.empty()) {
        return {};
    }

    const auto rowsOfGroup = [&](std::size_t g) { return Indices{ 2 * groups[g], 2 * groups[g] + 1 }; };

    const auto zOf = [&](std::size_t g) {
        const cv::Point2d& pixel = matches[groups[g]].value();
        return cv::Mat1d(cv::Vec2d(pixel.x, pixel.y), true);
    };

    const auto featureOf = [&](std::size_t g) { return o.features[groups[g]]; };

    // Low innovation inliers: the largest set that agrees with the update with a single match
    Indices inliers;
    double smallestResidual = 0.0;

    for (std::size_t hypothesis = 0; hypothesis < groups.size(); ++hypothesis) {
        const Indices rows = rowsOfGroup(hypothesis);

        const cv::Mat1d PHt = multiplyTransposed(o.Pcols, selectRows(o.H, rows));

        cv::Mat1d Kt;

        if (!cv::solve(selectRowsCols(o.S, rows, rows), PHt.t(), Kt, cv::DECOMP_CHOLESKY)) {
            continue;
        }

        const cv::Mat1d x(x_ + Kt.t() * (zOf(hypothesis) - selectRows(o.h, rows)));
        const Observation predicted = predictMeasurements(x);

        Indices agreeing;
        double residual = 0.0;

        for (std::size_t g = 0; g < groups.size(); ++g) {
            // A match whose feature the hypothesis does not predict in view cannot agree with it
            const std::optional<Indices> predictedRows = rowsOf(predicted, featureOf(g));

            if (!predictedRows.has_value()) {
                continue;
            }

            const double distance = cv::norm(zOf(g) - selectRows(predicted.h, predictedRows.value()));

            if (distance < maxDistance) {
                agreeing.push_back(g);
                residual += distance;
            }
        }

        if (agreeing.size() > inliers.size() || (agreeing.size() == inliers.size() && residual < smallestResidual)) {
            inliers = std::move(agreeing);
            smallestResidual = residual;
        }
    }

    if (inliers.empty()) {
        return {};
    }

    Indices rows;
    cv::Mat1d z;

    for (std::size_t g : inliers) {
        const Indices groupRows = rowsOfGroup(g);

        rows.insert(rows.end(), groupRows.begin(), groupRows.end());
        z.push_back(zOf(g));
    }

    correct(o, rows, z);

    // High innovation inliers: the rest of the matches that the updated state can explain
    const Observation updated = observe();

    Indices rescued;      // Groups
    Indices rescuedRows;  // Their rows in the updated observation
    cv::Mat1d rescuedZ;

    for (std::size_t g = 0; g < groups.size(); ++g) {
        if (std::find(inliers.begin(), inliers.end(), g) != inliers.end()) {
            continue;
        }

        const std::optional<Indices> updatedRows = rowsOf(updated, featureOf(g));

        if (!updatedRows.has_value()) {
            continue;
        }

        const cv::Mat1d innovation(zOf(g) - selectRows(updated.h, updatedRows.value()));

        cv::Mat1d weighted;
        cv::solve(selectRowsCols(updated.S, updatedRows.value(), updatedRows.value()),
                  innovation,
                  weighted,
                  cv::DECOMP_CHOLESKY);

        if (innovation.dot(weighted) < maxNIS) {
            rescued.push_back(g);
            rescuedRows.insert(rescuedRows.end(), updatedRows->begin(), updatedRows->end());
            rescuedZ.push_back(zOf(g));
        }
    }

    if (!rescuedRows.empty()) {
        correct(updated, rescuedRows, rescuedZ);
    }

    // The features of both stages, sorted
    std::vector<std::size_t> used;

    for (std::size_t g = 0; g < groups.size(); ++g) {
        if (std::find(inliers.begin(), inliers.end(), g) != inliers.end() ||
            std::find(rescued.begin(), rescued.end(), g) != rescued.end()) {
            used.push_back(featureOf(g));
        }
    }

    return used;
}

std::vector<std::size_t>
monoslam::EKF::measure(const std::vector<std::size_t>& features, const std::vector<cv::Point2d>& pixels, double maxNIS)
{
    const Observation o = observe();

    std::vector<std::size_t> measured;
    Indices rows;
    cv::Mat1d z;

    for (std::size_t k = 0; k < o.features.size(); ++k) {
        const auto it = std::find(features.begin(), features.end(), o.features[k]);

        if (it == features.end()) {
            continue;
        }

        const cv::Point2d& pixel = pixels[static_cast<std::size_t>(it - features.begin())];

        const Indices featureRows = { 2 * k, 2 * k + 1 };
        const cv::Vec2d innovation(pixel.x - o.h(static_cast<int>(2 * k)), pixel.y - o.h(static_cast<int>(2 * k + 1)));
        const cv::Matx22d S = selectRowsCols(o.S, featureRows, featureRows);

        if (innovation.dot(S.inv() * innovation) > maxNIS) {
            continue;
        }

        measured.push_back(o.features[k]);
        rows.insert(rows.end(), featureRows.begin(), featureRows.end());
        z.push_back(pixel.x);
        z.push_back(pixel.y);
    }

    if (!measured.empty()) {
        correct(o, rows, z);
    }

    return measured;
}

void
monoslam::EKF::addFeature(const cv::Point2d& pixel, double inverseDepth, double inverseDepthStd)
{
    const CameraPose pose = this->pose();

    // J is the Jacobian with respect to the camera position and orientation error, the pixel and
    // the inverse depth, in this order
    cv::Matx<double, featureSize, 9> J;
    const inverse_depth::Parameters y = inverse_depth::fromPixel(pose, camera_, pixel, inverseDepth, J);

    const cv::Matx66d Jpose = J.get_minor<featureSize, 6>(0, 0);
    const cv::Matx<double, featureSize, 3> Jnoise = J.get_minor<featureSize, 3>(0, 6);

    // Covariance of the pixel and the inverse depth
    const cv::Matx33d noise =
      cv::Matx33d::diag(cv::Vec3d(pixelStd_ * pixelStd_, pixelStd_ * pixelStd_, inverseDepthStd * inverseDepthStd));

    // Covariance of the new feature with the rest of the state, through the camera pose
    const cv::Mat1d crossCovariance(cv::Mat1d(Jpose) * P_.rowRange(0, 6));

    cv::Matx66d Ppose;
    P_(cv::Rect(0, 0, 6, 6)).copyTo(Ppose);

    const cv::Matx66d featureCovariance = Jpose * Ppose * Jpose.t() + Jnoise * noise * Jnoise.t();

    const int n = x_.rows;

    cv::Mat1d P(cv::Mat1d::zeros(n + featureSize, n + featureSize));

    P_.copyTo(P(cv::Rect(0, 0, n, n)));
    crossCovariance.copyTo(P(cv::Rect(0, n, n, featureSize)));
    cv::Mat1d(crossCovariance.t()).copyTo(P(cv::Rect(n, 0, featureSize, n)));
    cv::Mat1d(featureCovariance).copyTo(P(cv::Rect(n, n, featureSize, featureSize)));

    for (int k = 0; k < featureSize; ++k) {
        x_.push_back(y[k]);
    }

    P_ = P;
}

void
monoslam::EKF::removeFeatures(const std::vector<std::size_t>& features)
{
    if (features.empty()) {
        return;
    }

    Indices kept;

    for (int i = 0; i < x_.rows; ++i) {
        const bool camera = i < cameraStateSize;
        const auto feature = static_cast<std::size_t>((i - cameraStateSize) / featureSize);

        if (camera || std::find(features.begin(), features.end(), feature) == features.end()) {
            kept.push_back(static_cast<std::size_t>(i));
        }
    }

    x_ = selectRows(x_, kept);
    P_ = selectRowsCols(P_, kept, kept);
}

void
monoslam::EKF::resetCamera(const cv::Vec3d& r, const cv::Matx33d& R)
{
    constexpr double positionStd = 2.0;     // cm
    constexpr double orientationStd = 0.2;  // rad
    constexpr double linearVelStd = 30.0;   // cm/s
    constexpr double angularVelStd = 1.5;   // rad/s

    cv::Mat1d(r).copyTo(x_(cv::Rect(0, 0, 1, 3)));
    x_(cv::Rect(0, 3, 1, 9)).setTo(0.0);  // Orientation error and velocities
    R_ = R;

    // The new pose has nothing to do with the old one, and neither have their correlations
    P_.rowRange(0, cameraStateSize).setTo(0.0);
    P_.colRange(0, cameraStateSize).setTo(0.0);

    for (int i = 0; i < cameraStateSize; ++i) {
        const double std = i < 3 ? positionStd : i < 6 ? orientationStd : i < 9 ? linearVelStd : angularVelStd;
        P_(i, i) = std * std;
    }
}

bool
monoslam::EKF::isWellLocated(std::size_t i) const
{
    constexpr double maxRelativeStd = 0.1;

    const int rho = featureIndex(i) + 5;

    return x_(rho) > 0.0 && std::sqrt(P_(rho, rho)) < maxRelativeStd * x_(rho);
}

std::vector<cv::Point2f>
monoslam::EKF::project(const std::vector<cv::Point2f>& fallback) const
{
    const CameraPose pose = this->pose();

    std::vector<cv::Point2f> projections = fallback;

    for (std::size_t i = 0; i < projections.size(); ++i) {
        const std::optional<cv::Point2d> pixel = camera_.project(pose, featurePosition(x_, i));

        if (pixel.has_value()) {
            projections[i] = cv::Point2f(pixel.value());
        }
    }

    return projections;
}
