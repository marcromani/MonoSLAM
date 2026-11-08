#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "ekf.h"
#include "inverse_depth.h"
#include "relocalization.h"
#include "tracking.h"

namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();

// A camera with a strong lens distortion, as the real one, and a pinhole one
const cv::Matx33d K(600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0);
const monoslam::CameraModel camera(K, cv::Vec<double, 5>(-0.4, 0.3, 0.001, 0.001, -0.1), cv::Size(640, 480));
const monoslam::CameraModel pinhole(K, cv::Vec<double, 5>::all(0.0), cv::Size(640, 480));

const monoslam::MotionModel motion{ 100.0, 1.57, 0.5, 0.3 };

// Pixels with a standard deviation of 1, and every match accepted
const monoslam::MeasurementModel measurement{ 1.0, infinity, infinity };

const cv::Matx33d orientation = monoslam::rotation(cv::Vec3d(3.0, 0.1, -0.2));

cv::Mat1d
makeState()
{
    // Camera position, orientation error, linear and angular velocity
    const double cameraState[] = { 1.0, 2.0, 30.0, 0.0, 0.0, 0.0, 20.0, -10.0, 5.0, 0.8, -0.3, 1.2 };

    // Two features, seen first from other places
    const monoslam::inverse_depth::Parameters y1 =
      monoslam::inverse_depth::fromPoint(cv::Vec3d(-3.0, 1.0, 25.0), cv::Vec3d(0.0, 0.0, 0.0));
    const monoslam::inverse_depth::Parameters y2 =
      monoslam::inverse_depth::fromPoint(cv::Vec3d(4.0, 3.0, 28.0), cv::Vec3d(5.0, 2.0, 1.0));

    cv::Mat1d x(monoslam::cameraStateSize + 2 * monoslam::featureSize, 1);

    for (int i = 0; i < monoslam::cameraStateSize; ++i) {
        x(i) = cameraState[i];
    }

    for (int k = 0; k < monoslam::featureSize; ++k) {
        x(monoslam::featureIndex(0) + k) = y1[k];
        x(monoslam::featureIndex(1) + k) = y2[k];
    }

    return x;
}

/*
 * Compares an analytical Jacobian with central differences of a function, column by column.
 */
template<typename Function>
bool
matchesNumerical(const cv::Mat1d& analytical, const cv::Mat1d& at, Function f, double tolerance, const char* name)
{
    constexpr double step = 1e-6;

    for (int col = 0; col < at.rows; ++col) {
        cv::Mat1d plus = at.clone();
        cv::Mat1d minus = at.clone();

        plus(col) += step;
        minus(col) -= step;

        const cv::Mat1d numerical((f(plus) - f(minus)) / (2.0 * step));

        for (int row = 0; row < numerical.rows; ++row) {
            if (std::fabs(numerical(row) - analytical(row, col)) > tolerance) {
                std::cerr << name << '(' << row << ", " << col << ") = " << analytical(row, col) << ", expected "
                          << numerical(row) << '\n';
                return false;
            }
        }
    }

    return true;
}

cv::Mat1d
toMat(const monoslam::inverse_depth::Parameters& y)
{
    return cv::Mat1d(y, true);
}

/*
 * Returns the camera state predicted from one with an orientation error e, the orientation being
 * orientation exp(e), with its new error measured from a nominal predicted orientation. This is
 * what F and W are the derivatives of.
 */
cv::Mat1d
predictWithError(const cv::Mat1d& v, const cv::Matx33d& nominal, double dt)
{
    cv::Matx33d R = orientation * monoslam::rotation(monoslam::orientationError(v));

    cv::Mat1d x = v.rowRange(0, monoslam::cameraStateSize).clone();
    x.rowRange(3, 6).setTo(0.0);

    cv::Mat1d xNew = motion.predict(x, R, dt);

    // The error is tiny, and for a tiny rotation exp(e) = I + [e]x: cv::Rodrigues rounds it to zero
    const cv::Matx33d E = nominal.t() * R;
    cv::Mat1d(cv::Vec3d(E(2, 1) - E(1, 2), E(0, 2) - E(2, 0), E(1, 0) - E(0, 1)) * 0.5).copyTo(xNew.rowRange(3, 6));

    return xNew;
}

// H must match the numerical derivative everywhere, including the columns of other features
bool
testObservationJacobian()
{
    // The predicted measurements of a filter at a state
    const auto observeAt = [&](const cv::Mat1d& at) {
        const cv::Mat1d P(cv::Mat1d::eye(at.rows, at.rows));
        const monoslam::EKF ekf(motion, measurement, camera, at, orientation, P);
        return ekf.observe();
    };

    const cv::Mat1d x = makeState();
    const monoslam::Observation o = observeAt(x);

    if (o.h.rows != 4) {
        std::cerr << "The features should be visible\n";
        return false;
    }

    return matchesNumerical(o.H, x, [&](const cv::Mat1d& v) { return observeAt(v).h; }, 1e-4, "H");
}

// F must match the numerical derivative of the prediction with respect to the camera state
bool
testProcessJacobian()
{
    constexpr double dt = 1.0 / 30.0;

    const cv::Mat1d x = makeState().rowRange(0, monoslam::cameraStateSize).clone();

    cv::Matx33d nominal = orientation;
    cv::Matx<double, monoslam::cameraStateSize, monoslam::cameraStateSize> F;
    motion.predict(x, nominal, dt, &F);

    return matchesNumerical(
      cv::Mat1d(F), x, [&](const cv::Mat1d& v) { return predictWithError(v, nominal, dt); }, 1e-6, "F");
}

// W must move the orientation like an angular acceleration that is constant during the step,
// which adds 0.5 * big_w * dt to the rotation, w_old * h, where h is the duration of the decay
bool
testProcessNoiseJacobian()
{
    constexpr double dt = 1.0 / 30.0;

    const cv::Mat1d x = makeState().rowRange(0, monoslam::cameraStateSize).clone();

    cv::Matx33d nominal = orientation;
    cv::Matx<double, monoslam::cameraStateSize, 6> W;
    motion.predict(x, nominal, dt, nullptr, &W);

    const double h = motion.angularDecayTime * (1.0 - std::exp(-dt / motion.angularDecayTime));

    const auto f = [&](const cv::Mat1d& bigW) {
        cv::Mat1d v = x.clone();
        v.rowRange(9, 12) += bigW * (0.5 * dt / h);
        return cv::Mat1d(predictWithError(v, nominal, dt).rowRange(3, 6));
    };

    return matchesNumerical(cv::Mat1d(W.get_minor<3, 3>(3, 3)), cv::Mat1d::zeros(3, 1), f, 1e-6, "W");
}

// The features do not move and their uncertainty does not grow during the prediction
bool
testPredictionKeepsFeatures()
{
    const cv::Mat1d x = makeState();
    const cv::Mat1d P(0.01 * cv::Mat1d::eye(x.rows, x.rows));

    monoslam::EKF ekf(motion, measurement, pinhole, x, orientation, P);
    ekf.predict(1.0 / 30.0);

    const int first = monoslam::cameraStateSize;
    const int numFeatureStates = x.rows - first;
    const cv::Rect features(first, first, numFeatureStates, numFeatureStates);

    const double featureChange = cv::norm(ekf.state().rowRange(first, x.rows), x.rowRange(first, x.rows));
    const double featureCovarianceChange = cv::norm(ekf.covariance()(features), P(features));

    return featureChange < 1e-12 && featureCovarianceChange < 1e-12 && ekf.covariance()(0, 0) > P(0, 0);
}

// unproject must invert the projection, including the lens distortion
bool
testUnproject()
{
    const cv::Vec3d ray = camera.unproject(camera.distort(0.3, -0.2));

    return cv::norm(ray - cv::Vec3d(0.3, -0.2, 1.0)) < 1e-4;
}

// Two regions moving apart, as two surfaces at different depths do, cannot be explained by a
// single plane: each seed has to follow the motion of its own surroundings
bool
testFlowFollowsDepthLayers()
{
    cv::Mat1b texture(480, 640);
    cv::theRNG().state = 12345;
    cv::randu(texture, 0, 255);
    cv::GaussianBlur(texture, texture, cv::Size(0, 0), 2.5);

    const int leftShift = 8;
    const int rightShift = -6;

    cv::Mat1b next = texture.clone();

    texture(cv::Rect(0, 0, 300, 480)).copyTo(next(cv::Rect(leftShift, 0, 300, 480)));
    texture(cv::Rect(340, 0, 300, 480)).copyTo(next(cv::Rect(340 + rightShift, 0, 300, 480)));

    monoslam::FlowTracker tracker(texture, { cv::Point2f(150.0f, 240.0f), cv::Point2f(490.0f, 240.0f) });

    const std::vector<cv::Point2f>& seeds = tracker.predict(next);

    return std::fabs(seeds[0].x - (150.0f + leftShift)) < 1.0f && std::fabs(seeds[1].x - (490.0f + rightShift)) < 1.0f;
}

// The Jacobian of the distortion must match the numerical derivative of distort
bool
testDistortionJacobian()
{
    // Unequal focal lengths and tangential coefficients, so that a swap of x and y shows
    const cv::Matx33d Kxy(600.0, 0.0, 320.0, 0.0, 610.0, 240.0, 0.0, 0.0, 1.0);
    const monoslam::CameraModel asymmetric(Kxy, cv::Vec<double, 5>(-0.4, 0.3, 0.002, -0.001, -0.1), cv::Size(640, 480));

    const cv::Mat1d at(cv::Vec2d(0.3, -0.2));

    return matchesNumerical(
      cv::Mat1d(asymmetric.distortionJacobian(at(0), at(1))),
      at,
      [&](const cv::Mat1d& v) {
          const cv::Point2d pixel = asymmetric.distort(v(0), v(1));
          return cv::Mat1d(cv::Vec2d(pixel.x, pixel.y));
      },
      1e-4,
      "distortion");
}

// fromPoint and toPoint must be inverse, and fromPointJacobian the derivative of fromPoint
bool
testInverseDepthPoint()
{
    namespace id = monoslam::inverse_depth;

    const cv::Vec3d anchor(1.0, -2.0, 3.0);
    const cv::Vec3d point(4.0, 5.0, 60.0);

    if (cv::norm(id::toPoint(id::fromPoint(anchor, point)) - point) > 1e-9) {
        std::cerr << "toPoint(fromPoint(point)) is not the point\n";
        return false;
    }

    const cv::Mat1d at(point, true);

    return matchesNumerical(
      cv::Mat1d(id::fromPointJacobian(anchor, point)),
      at,
      [&](const cv::Mat1d& p) {
          const id::Parameters y = id::fromPoint(anchor, cv::Vec3d(p(0), p(1), p(2)));
          return cv::Mat1d(cv::Vec3d(y[3], y[4], y[5]));
      },
      1e-6,
      "fromPoint");
}

// The Jacobian of fromPixel must match the numerical derivative with respect to the camera
// position and orientation error, the pixel and the inverse depth
bool
testInitializationJacobian()
{
    namespace id = monoslam::inverse_depth;

    const cv::Matx33d R = monoslam::rotation(cv::Vec3d(0.3, -0.2, 0.1));

    // Camera position, orientation error, pixel and inverse depth
    const cv::Mat1d at(cv::Vec<double, 9>(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 400.0, 150.0, 0.02));

    const auto f = [&](const cv::Mat1d& v) {
        const monoslam::CameraPose pose(cv::Vec3d(v(0), v(1), v(2)),
                                        R * monoslam::rotation(cv::Vec3d(v(3), v(4), v(5))));

        cv::Matx<double, id::size, 9> J;
        return toMat(id::fromPixel(pose, camera, cv::Point2d(v(6), v(7)), v(8), J));
    };

    const monoslam::CameraPose pose(cv::Vec3d(1.0, 2.0, 3.0), R);

    cv::Matx<double, id::size, 9> J;
    id::fromPixel(pose, camera, cv::Point2d(400.0, 150.0), 0.02, J);

    // The ray of a pixel is found iteratively, so its numerical derivative is less precise
    return matchesNumerical(cv::Mat1d(J), at, f, 1e-5, "fromPixel");
}

/*
 * Runs the filter on a synthetic sequence: the camera moves sideways at a known speed, looking at
 * a single point, which is added from its first pixel with a wrong inverse depth. Returns the final
 * state and covariance.
 *
 * point           True position of the point
 * inverseDepth    Initial inverse depth of the feature
 * x               Final state
 * P               Final covariance
 */
void
runSyntheticSequence(const cv::Vec3d& point, double inverseDepth, cv::Mat1d& x, cv::Mat1d& P)
{
    constexpr double dt = 1.0 / 30.0;
    const cv::Vec3d velocity(30.0, 0.0, 0.0);  // cm/s

    // The camera looks along the z axis, and its motion is known
    x = cv::Mat1d::zeros(monoslam::cameraStateSize, 1);
    x(6) = velocity[0];

    P = 1e-10 * cv::Mat1d::eye(x.rows, x.rows);

    const auto project = [&](const cv::Vec3d& position) {
        const monoslam::CameraPose pose(position, cv::Matx33d::eye());
        return pinhole.project(pose, point).value();
    };

    // A motion model that knows the velocity, so that only the depth is to be found
    const monoslam::MotionModel knownMotion{ 1e-3, 1e-3, 1e6, 1e6 };

    monoslam::EKF ekf(knownMotion, measurement, pinhole, x, cv::Matx33d::eye(), P);
    ekf.addFeature(project(cv::Vec3d::all(0.0)), inverseDepth, 0.02);

    for (int k = 1; k <= 60; ++k) {
        ekf.predict(dt);
        ekf.update(ekf.observe(), { project(velocity * dt * k) }, { cv::Matx22d::eye() });
    }

    x = ekf.state().clone();
    P = ekf.covariance().clone();
}

// A feature added with a wrong depth must converge to the true one as the camera translates
bool
testDepthConvergence()
{
    const cv::Vec3d point(30.0, -3.0, 100.0);  // In view from x = 0 to 60 cm

    cv::Mat1d x, P;
    runSyntheticSequence(point, 1.0 / 40.0, x, P);  // Initially 40 cm away instead of 100

    const int rho = monoslam::featureIndex(0) + 5;

    const double distance = cv::norm(monoslam::featurePosition(x, 0));
    const double inverseDepthStd = std::sqrt(P(rho, rho));

    if (std::fabs(distance - cv::norm(point)) > 0.02 * cv::norm(point) || inverseDepthStd > 0.002) {
        std::cerr << "Distance " << distance << " instead of " << cv::norm(point) << ", inverse depth std "
                  << inverseDepthStd << '\n';
        return false;
    }

    return true;
}

// A point at infinity must keep an inverse depth close to zero, without numerical trouble
bool
testPointAtInfinity()
{
    const cv::Vec3d point(1e8, -5e7, 1e9);  // As good as infinitely far

    cv::Mat1d x, P;
    runSyntheticSequence(point, 1.0 / 40.0, x, P);

    const double inverseDepth = x(monoslam::featureIndex(0) + 5);

    if (!cv::checkRange(x) || !cv::checkRange(P) || std::fabs(inverseDepth) > 1e-3) {
        std::cerr << "Inverse depth " << inverseDepth << " for a point at infinity\n";
        return false;
    }

    return true;
}

// The descriptor of a point must survive a rotation of the image, and differ from that of another point
bool
testDescriptorRotation()
{
    cv::Mat1b image(240, 240);
    cv::RNG rng(7);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(0, 0), 2.0);

    const cv::Point2d center(120.0, 120.0);
    const cv::Point2d other(80.0, 150.0);

    cv::Mat1b rotated;
    cv::warpAffine(image, rotated, cv::getRotationMatrix2D(cv::Point2f(center), 60.0, 1.0), image.size());

    const cv::Mat1b original = monoslam::describe(image, center);
    const cv::Mat1b turned = monoslam::describe(rotated, center);
    const cv::Mat1b elsewhere = monoslam::describe(image, other);

    if (original.empty() || turned.empty() || elsewhere.empty()) {
        std::cerr << "Missing descriptor\n";
        return false;
    }

    const double same = cv::norm(original, turned, cv::NORM_HAMMING);
    const double different = cv::norm(original, elsewhere, cv::NORM_HAMMING);

    if (same > 40.0 || different < 2.0 * same) {
        std::cerr << "Hamming distance " << same << " after the rotation, " << different << " to another point\n";
        return false;
    }

    return true;
}

}

int
main()
{
    bool passed = true;

    const auto run = [&](const char* name, bool (*test)()) {
        const bool result = test();

        std::cout << (result ? "[PASS] " : "[FAIL] ") << name << '\n';

        passed = passed && result;
    };

    run("Observation Jacobian", testObservationJacobian);
    run("Process Jacobian", testProcessJacobian);
    run("Process noise Jacobian", testProcessNoiseJacobian);
    run("Prediction keeps features", testPredictionKeepsFeatures);
    run("Unproject", testUnproject);
    run("Distortion Jacobian", testDistortionJacobian);
    run("Inverse depth point", testInverseDepthPoint);
    run("Initialization Jacobian", testInitializationJacobian);
    run("Depth convergence", testDepthConvergence);
    run("Point at infinity", testPointAtInfinity);
    run("Flow follows depth layers", testFlowFollowsDepthLayers);
    run("Descriptor rotation", testDescriptorRotation);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
