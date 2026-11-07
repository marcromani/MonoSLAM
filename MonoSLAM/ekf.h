#ifndef MONOSLAM_EKF_H
#define MONOSLAM_EKF_H

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "inverse_depth.h"

namespace monoslam {

/*
 * The state is [r, e, v, w, y_1, ..., y_N]: camera position (3), orientation error (3), linear
 * velocity (3), angular velocity in camera coordinates (3), and the inverse depth parameters of
 * every feature (6 each, see inverse_depth.h). The functions below are the only places that need to
 * know this layout.
 *
 * The orientation itself is a rotation matrix R, from camera to world, kept by the filter next to
 * the state (an error-state, or multiplicative, EKF). The state only holds a small rotation e of the
 * camera on top of it, the true orientation being R exp(e), and P its uncertainty. After every
 * update R absorbs e, which goes back to zero. Unlike a quaternion, e has as many entries as the
 * orientation has degrees of freedom, so there is no unit norm to keep, and unlike a rotation vector
 * of the whole orientation, it stays small, far from the angle of pi where that wraps around.
 */
inline constexpr int cameraStateSize = 12;
inline constexpr int featureSize = inverse_depth::size;

inline cv::Vec3d
cameraPosition(const cv::Mat1d& x)
{
    return cv::Vec3d(x(0), x(1), x(2));
}

inline cv::Vec3d
orientationError(const cv::Mat1d& x)
{
    return cv::Vec3d(x(3), x(4), x(5));
}

/*
 * Returns the index in the state of the first parameter of feature i.
 */
inline int
featureIndex(std::size_t i)
{
    return cameraStateSize + featureSize * static_cast<int>(i);
}

inline inverse_depth::Parameters
featureParameters(const cv::Mat1d& x, std::size_t i)
{
    inverse_depth::Parameters y;

    for (int k = 0; k < featureSize; ++k) {
        y[k] = x(featureIndex(i) + k);
    }

    return y;
}

/*
 * Returns the position of feature i in world coordinates, see inverse_depth::toPoint.
 */
inline cv::Vec3d
featurePosition(const cv::Mat1d& x, std::size_t i)
{
    return inverse_depth::toPoint(featureParameters(x, i));
}

inline std::size_t
numFeatures(const cv::Mat1d& x)
{
    return static_cast<std::size_t>((x.rows - cameraStateSize) / featureSize);
}

/*
 * Constant velocity model whose velocities decay towards zero. Between two frames the camera
 * undergoes a random, constant acceleration, and its linear and angular velocities decay with time
 * constants tau, v_new = exp(-dt / tau) v_old + big_v. A hand-held camera keeps moving from one
 * frame to the next, but not for long in the same way, and the decay stops the prediction from
 * drifting away when nothing is measured. For tau -> infinity this is the plain constant velocity
 * model of Davison et al.
 */
struct MotionModel
{
    double linearAccStd;      // Standard deviation of the linear acceleration
    double angularAccStd;     // Standard deviation of the angular acceleration
    double linearDecayTime;   // Time constant of the decay of the linear velocity
    double angularDecayTime;  // Time constant of the decay of the angular velocity

    /*
     * Returns the camera part of the state after dt, moves the orientation, and optionally returns
     * the Jacobians of the new camera state with respect to the old one, F (12 x 12), and to the
     * noise, the velocity changes big_v and big_w, W (12 x 6). The orientation error must be zero.
     *
     * x     State, of which only the camera part is used
     * R     Orientation, from camera to world
     * dt    Time step
     * F     Jacobian with respect to the camera state
     * W     Jacobian with respect to the noise
     */
    cv::Mat1d predict(const cv::Mat1d& x,
                      cv::Matx33d& R,
                      double dt,
                      cv::Matx<double, cameraStateSize, cameraStateSize>* F = nullptr,
                      cv::Matx<double, cameraStateSize, 6>* W = nullptr) const;

    /*
     * Returns the covariance of the noise during dt: the velocity changes of a constant acceleration.
     */
    cv::Matx66d noiseCovariance(double dt) const;
};

/*
 * The features predicted in view from a state, and what the update needs about them.
 */
struct Observation
{
    std::vector<std::size_t> features;  // Features in view

    cv::Mat1d h;  // Their predicted pixels, two rows per feature
    cv::Mat1d H;  // Jacobian of h, with the columns of the camera and then 6 per feature in view
    cv::Mat1d S;  // Innovation covariance, H P H^T plus the pixel noise

    cv::Mat1d Pcols;  // The columns of P that H acts on
};

/*
 * The EKF of the camera and the map. The camera moves with the motion model, the features are
 * static, and the measurements are the pixels where the features are found.
 */
class EKF
{
  public:
    /*
     * motion      Motion model
     * camera      Camera model
     * pixelStd    Standard deviation of a pixel measurement
     * x           Initial state
     * R           Initial orientation, from camera to world
     * P           Initial covariance
     */
    EKF(const MotionModel& motion,
        const CameraModel& camera,
        double pixelStd,
        const cv::Mat1d& x,
        const cv::Matx33d& R,
        const cv::Mat1d& P);

    const cv::Mat1d& state() const { return x_; }
    const cv::Matx33d& orientation() const { return R_; }
    const cv::Mat1d& covariance() const { return P_; }

    /*
     * Returns the camera pose.
     */
    CameraPose pose() const { return CameraPose(cameraPosition(x_), R_); }

    /*
     * Moves the camera dt into the future. The features do not move, so only the camera block of
     * P and its correlations with the features change.
     */
    void predict(double dt);

    /*
     * Returns the features in view from the current state and their predicted measurements.
     */
    Observation observe() const;

    /*
     * Updates the state with the matches of the features in view, and returns the features whose
     * matches were used. Wrong matches are left out with 1-point RANSAC (Civera et al., 2010): every
     * match proposes the state that the update with it alone would give, the matches within
     * maxDistance pixels of their prediction from it support it, and the state is updated with the
     * largest support. Then the matches that the updated state explains, with a normalized
     * innovation squared below maxNIS, are rescued and used in a second update.
     *
     * o              Observation, from observe
     * matches        Pixel of each feature of o, if it was found
     * noises         Covariance of each match, or nothing for the pixel noise of the constructor
     * maxDistance    Largest distance of a supporting match, in pixels
     * maxNIS         Largest normalized innovation squared of a rescued match
     */
    std::vector<std::size_t> update(Observation o,
                                    const std::vector<std::optional<cv::Point2d>>& matches,
                                    const std::vector<cv::Matx22d>& noises,
                                    double maxDistance,
                                    double maxNIS);

    /*
     * Adds a feature first seen at a pixel. Its anchor and ray come from the camera pose and the
     * pixel, so they inherit the uncertainty of both, and its correlation with the camera. Its
     * inverse depth is a guess with a large uncertainty, which the following updates refine as the
     * camera moves.
     *
     * pixel              Pixel where the feature is seen
     * inverseDepth       Initial inverse depth
     * inverseDepthStd    Its standard deviation
     */
    void addFeature(const cv::Point2d& pixel, double inverseDepth, double inverseDepthStd);

    /*
     * Removes features from the state.
     *
     * features    Sorted indices of the features to remove
     */
    void removeFeatures(const std::vector<std::size_t>& features);

    /*
     * Moves the camera to a pose found without the filter. The features keep their estimates and
     * their uncertainty, the camera gets a large uncertainty, without correlations with the
     * features, and no velocity, since nothing is known about the motion that took it there.
     *
     * r    Camera position
     * R    Camera orientation, from camera to world
     */
    void resetCamera(const cv::Vec3d& r, const cv::Matx33d& R);

    /*
     * Returns whether the position of a feature is known well enough to treat it as exact, as PnP
     * does: its distance must be known within 10%.
     */
    bool isWellLocated(std::size_t i) const;

    /*
     * Returns the projections of all the features, or a fallback for those out of view.
     *
     * fallback    Positions to use for the features out of view
     */
    std::vector<cv::Point2f> project(const std::vector<cv::Point2f>& fallback) const;

  private:
    /*
     * Returns the features in view from a state, their predicted measurements and Jacobian. The
     * orientation is R exp(e), with the error e of the state.
     */
    Observation predictMeasurements(const cv::Mat1d& x) const;

    /*
     * Gives some features of an observation their own measurement noise in S, instead of the pixel
     * noise of the constructor.
     *
     * o           Observation
     * features    Features
     * noises      Covariance of the measurement of each feature
     */
    void setNoises(Observation& o,
                   const std::vector<std::size_t>& features,
                   const std::vector<cv::Matx22d>& noises) const;

    /*
     * Applies the Kalman update with some of the rows of an observation.
     *
     * o       Observation
     * rows    Sorted rows of o.h that were measured
     * z       Measurements of those rows
     */
    void correct(const Observation& o, const std::vector<std::size_t>& rows, const cv::Mat1d& z);

    MotionModel motion_;
    CameraModel camera_;
    double pixelStd_;

    cv::Mat1d x_;
    cv::Matx33d R_;
    cv::Mat1d P_;
};

}

#endif
