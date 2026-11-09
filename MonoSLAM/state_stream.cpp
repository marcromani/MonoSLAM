#include "state_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inverse_depth.h"

namespace {

constexpr int jpegQuality = 80;

/*
 * Returns the base64 encoding of some bytes.
 */
std::string
base64(const unsigned char* bytes, std::size_t size)
{
    static const char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string text;

    for (std::size_t i = 0; i < size; i += 3) {
        const std::size_t n = std::min<std::size_t>(3, size - i);

        std::uint32_t group = 0;

        for (std::size_t k = 0; k < n; ++k) {
            group |= static_cast<std::uint32_t>(bytes[i + k]) << (16 - 8 * k);
        }

        for (std::size_t k = 0; k < 4; ++k) {
            text += k <= n ? digits[(group >> (18 - 6 * k)) & 63] : '=';
        }
    }

    return text;
}

/*
 * Returns where a pixel of the frame is in the undistorted frame, and the Jacobian of that mapping there.
 *
 * camera    Camera model
 * pixel     Pixel of the frame
 * J         Jacobian of the undistorted pixel with respect to the pixel
 */
cv::Point2d
undistort(const monoslam::CameraModel& camera, const cv::Point2d& pixel, cv::Matx22d& J)
{
    const cv::Matx33d& K = camera.K();
    const cv::Vec3d ray = camera.unproject(pixel);

    J = cv::Matx22d(K(0, 0), 0.0, 0.0, K(1, 1)) * camera.distortionJacobian(ray[0], ray[1]).inv();

    return { K(0, 0) * ray[0] + K(0, 2), K(1, 1) * ray[1] + K(1, 2) };
}

}

monoslam::StateStream::StateStream(const CameraModel& camera)
  : camera_(camera)
{
    cv::initUndistortRectifyMap(
      camera.K(), camera.distCoeffs(), cv::Matx33d::eye(), camera.K(), camera.frameSize(), CV_16SC2, mapX_, mapY_);
}

bool
monoslam::StateStream::open(const std::string& path)
{
    file_.open(path);

    if (!file_.is_open()) {
        return false;
    }

    const cv::Matx33d& K = camera_.K();

    file_ << "camera " << camera_.frameSize().width << ' ' << camera_.frameSize().height << ' ' << K(0, 0) << ' '
          << K(1, 1) << ' ' << K(0, 2) << ' ' << K(1, 2) << '\n';

    return true;
}

void
monoslam::StateStream::write(std::size_t frame,
                             double time,
                             const cv::Mat1b& image,
                             const EKF& ekf,
                             const std::vector<Feature>& features,
                             const std::vector<std::size_t>& inView,
                             const std::vector<std::size_t>& used,
                             const std::vector<DrawnMatch>& matches)
{
    const cv::Mat1d& x = ekf.state();
    const cv::Mat1d& P = ekf.covariance();

    const cv::Vec3d r = cameraPosition(x);

    cv::Vec3d o;
    cv::Rodrigues(ekf.orientation(), o);

    cv::Matx33d positionCovariance;
    P(cv::Rect(0, 0, 3, 3)).copyTo(positionCovariance);

    cv::Vec3d eigenvalues;
    cv::eigen(positionCovariance, eigenvalues);

    // The time with all its digits, to the microsecond
    file_ << "frame " << frame << ' ' << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << o[0] << ' ' << o[1] << ' ' << o[2]
          << ' ' << used.size() << ' ' << std::sqrt(eigenvalues[0]) << ' ' << std::to_string(time) << '\n';

    cv::Mat1b undistorted;
    cv::remap(image, undistorted, mapX_, mapY_, cv::INTER_LINEAR);

    std::vector<unsigned char> jpeg;
    cv::imencode(".jpg", undistorted, jpeg, { cv::IMWRITE_JPEG_QUALITY, jpegQuality });

    file_ << "image " << base64(jpeg.data(), jpeg.size()) << '\n';

    for (const Feature& feature : features) {
        if (feature.id >= nextPatch_) {
            const cv::Mat1b patch = feature.frame(feature.roi).clone();

            file_ << "patch " << feature.id << ' ' << Feature::patchSize << ' ' << base64(patch.data, patch.total())
                  << '\n';

            nextPatch_ = feature.id + 1;
        }
    }

    for (std::size_t i = 0; i < features.size(); ++i) {
        const inverse_depth::Parameters y = featureParameters(x, i);

        const int rho = featureIndex(i) + 5;
        const double rhoStd = std::sqrt(P(rho, rho));

        // The ray within two standard deviations: the far end is at infinity if rho can be zero
        inverse_depth::Parameters nearest = y;
        inverse_depth::Parameters farthest = y;
        nearest[5] = y[5] + 2.0 * rhoStd;
        farthest[5] = y[5] - 2.0 * rhoStd;

        const cv::Vec3d p = inverse_depth::toPoint(y);
        const cv::Vec3d near = inverse_depth::toPoint(nearest);
        const cv::Vec3d far = inverse_depth::toPoint(farthest);

        int status = 0;

        if (std::find(used.begin(), used.end(), i) != used.end()) {
            status = 2;

        } else if (std::find(inView.begin(), inView.end(), i) != inView.end()) {
            status = 1;
        }

        file_ << "feature " << features[i].id << ' ' << status << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << ' '
              << near[0] << ' ' << near[1] << ' ' << near[2] << ' ' << far[0] << ' ' << far[1] << ' ' << far[2];

        for (const cv::Vec3d& corner : features[i].patchCorners(camera_, p)) {
            file_ << ' ' << corner[0] << ' ' << corner[1] << ' ' << corner[2];
        }

        file_ << '\n';
    }

    for (const DrawnMatch& d : matches) {
        cv::Matx22d J, unused;
        const cv::Point2d prediction = undistort(camera_, d.prediction, J);
        const cv::Point2d seed = undistort(camera_, d.seed, unused);
        const cv::Point2d match = undistort(camera_, d.match, unused);

        // The matrix of the ellipse, from its axes, and in the undistorted frame
        const double angle = d.ellipse.angle * CV_PI / 180.0;
        const cv::Matx22d axes(std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle));
        const double a = d.ellipse.size.width / 2.0;
        const double b = d.ellipse.size.height / 2.0;
        const cv::Matx22d E = J * axes * cv::Matx22d(a * a, 0.0, 0.0, b * b) * axes.t() * J.t();

        file_ << "match " << prediction.x << ' ' << prediction.y << ' ' << seed.x << ' ' << seed.y << ' ' << match.x
              << ' ' << match.y << ' ' << E(0, 0) << ' ' << E(0, 1) << ' ' << E(1, 1) << '\n';
    }

    file_ << "end" << std::endl;  // Flushes
}
