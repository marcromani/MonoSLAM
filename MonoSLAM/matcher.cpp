#include "matcher.h"

#include <opencv2/imgproc.hpp>

#include "filter/observation_model.h"
#include "geometry.h"
#include "quaternion.h"
#include "util.h"

monoslam::Matcher::Matcher() {}

monoslam::filter::Measurement
monoslam::Matcher::match(const cv::Mat1b& frame,
                         const std::vector<Feature>& features,
                         const cv::Matx33d& K,
                         const cv::Mat1d& x,
                         const filter::ObservationPrediction& p,
                         const cv::Mat1d& S)
{
    assert(!p.h.empty());

    std::vector<cv::Point2d> z;
    z.reserve(static_cast<size_t>(p.h.rows / 2));

    std::vector<size_t> measured;
    measured.reserve(static_cast<size_t>(p.h.rows));

    const std::vector<cv::RotatedRect> ellipses = util::computeEllipses(p.h.reshape(2, p.h.rows / 2), S);

    const bool fullWidth = p.active.empty();

    for (int i = 7; i < p.H.cols; i += 3) {
        const size_t activeIdx = static_cast<size_t>(i);                           // Index in p.active
        const size_t globalIdx = fullWidth ? activeIdx + 6 : p.active[activeIdx];  // Index in the state x
        const size_t featureIdx = (globalIdx - 13) / 3;                            // Feature index

        if (featureIdx >= ellipses.size()) {
            continue;
        }

        const cv::Vec3d position(
          x.rowRange(static_cast<int>(globalIdx), static_cast<int>(globalIdx) + 3));  // Current feature position

        const cv::Vec3d t(x(0), x(1), x(2));                                            // Current camera position
        const cv::Matx33d R = quaternion::toMatrix(cv::Vec4d(x(3), x(4), x(5), x(6)));  // Current camera rotation

        const Feature& feature = features[featureIdx];

        int u, v;
        cv::Mat1b mask;

        const cv::Mat1b warped = geometry::warpPatch(
          position, feature.normal, feature.frame, feature.roi, K, feature.R, feature.t, R.t(), t, u, v, mask);

        if (warped.empty()) {
            // TODO

        } else {
            cv::Rect roi = util::getBoundingBox(ellipses[featureIdx], frame.size());

            const cv::Mat1b extendedFrame = util::computeMatchingImage(frame, warped, u, v, roi);

            cv::Mat1f result;
            cv::matchTemplate(extendedFrame, warped, result, cv::TM_CCORR_NORMED, mask);

            double value;
            cv::Point location;
            cv::minMaxLoc(result, 0, &value, 0, &location);

            if (value > 0.9) {
                const int px = location.x + roi.x + u;
                const int py = location.y + roi.y + v;

                z.emplace_back(px, py);

                measured.push_back(2 * (activeIdx - 7) / 3);
                measured.push_back(2 * (activeIdx - 7) / 3 + 1);

            } else {
            }
        }
    }

    if (z.empty()) {
        return {};
    }

    filter::Measurement measurement;

    measurement.z = cv::Mat1d(static_cast<int>(2 * z.size()), 1);

    for (int i = 0; i < measurement.z.rows / 2; i += 2) {
        measurement.z(i) = z[static_cast<size_t>(i)].x;
        measurement.z(i + 1) = z[static_cast<size_t>(i)].y;
    }

    measurement.measured = measured;

    return measurement;
}
