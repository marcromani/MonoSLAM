#include "matcher.h"

#include <opencv2/imgproc.hpp>

#include "filter/observation_model.h"
#include "geometry.h"
#include "logging.h"
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

    for (size_t i = 7; i < static_cast<size_t>(p.H.cols); i += 3) {
        const size_t hIdx = (i - 7) / 3;                      // Index of the feature as a row in h
        const size_t xIdx = fullWidth ? i + 6 : p.active[i];  // Index of the feature as a row in x
        const size_t fIdx = (xIdx - 13) / 3;                  // Index of the feature

        assert(hIdx < ellipses.size());

        const cv::Vec3d position(
          x.rowRange(static_cast<int>(xIdx), static_cast<int>(xIdx) + 3));  // Current feature position

        const cv::Vec3d t(x(0), x(1), x(2));                                            // Current camera position
        const cv::Matx33d R = quaternion::toMatrix(cv::Vec4d(x(3), x(4), x(5), x(6)));  // Current camera rotation

        const Feature& feature = features[fIdx];

        cv::Point2d center;
        cv::Mat1b mask;

        const cv::Mat1b warped = geometry::warpPatch(position,
                                                     feature.normal,
                                                     feature.frame,
                                                     feature.position,
                                                     feature.roi,
                                                     K,
                                                     feature.R,
                                                     feature.t,
                                                     R.t(),
                                                     t,
                                                     center,
                                                     mask);

        if (warped.empty()) {
            // TODO

        } else {
            cv::Rect roi = util::getBoundingBox(ellipses[hIdx], frame.size());

            roi &= cv::Rect(0, 0, frame.cols, frame.rows);

            if (roi.empty() || roi.width <= 0 || roi.height <= 0) {
                const int idx = static_cast<int>(2 * hIdx);
                const cv::Mat1d Si = S(cv::Rect(idx, idx, 2, 2));

                MONOSLAM_LOG(monoslam::logging::LogLevel::DEBUG,
                             "[Empty matching ROI]\n",
                             "          Feature index: ",
                             fIdx,
                             '\n',
                             "  Predicted measurement:",
                             p.h.rowRange(idx, idx + 2).t(),
                             "  Innovation covariance:",
                             Si,
                             "         Ellipse centre: ",
                             ellipses[hIdx].center,
                             '\n',
                             "           Ellipse size: ",
                             ellipses[hIdx].size,
                             '\n',
                             "             Frame size: ",
                             frame.size(),
                             '\n');
                continue;
            }

            cv::Point origin;
            const cv::Mat1b extendedFrame = util::computeMatchingImage(frame, warped, center.x, center.y, roi, origin);

            cv::Mat1f result;
            cv::matchTemplate(extendedFrame, warped, result, cv::TM_CCORR_NORMED, mask);

            double value;
            cv::Point location;
            cv::minMaxLoc(result, 0, &value, 0, &location);

            if (value > 0.9) {
                const double px = origin.x + location.x + center.x;
                const double py = origin.y + location.y + center.y;

                const int idx = static_cast<int>(2 * hIdx);
                const cv::Matx22d Si = S(cv::Rect(idx, idx, 2, 2));

                const cv::Vec2d innovation = cv::Vec2d(px - p.h(idx), py - p.h(idx + 1));

                const double NIS = innovation.dot(Si.inv() * innovation);

                if (!(NIS < 9.210340371976184)) {
                    continue;
                }

                z.emplace_back(px, py);

                measured.push_back(2 * hIdx);
                measured.push_back(2 * hIdx + 1);

                MONOSLAM_LOG(monoslam::logging::LogLevel::DEBUG,
                             "[Match] Feature ",
                             fIdx,
                             '\n',
                             "     Prediction: (",
                             p.h(static_cast<int>(2 * hIdx)),
                             ", ",
                             p.h(static_cast<int>(2 * hIdx + 1)),
                             ")\n",
                             "     Matched at: (",
                             px,
                             ", ",
                             py,
                             ")\n",
                             "          Score: ",
                             value,
                             '\n',
                             "     Patch size: ",
                             warped.size(),
                             '\n',
                             "     Frame size: ",
                             extendedFrame.size(),
                             '\n',
                             "  Masked pixels: ",
                             cv::countNonZero(mask),
                             '/',
                             mask.total(),
                             '\n',
                             "            ROI: ",
                             roi,
                             '\n');

            } else {
                // TODO
            }
        }
    }

    if (z.empty()) {
        return {};
    }

    filter::Measurement measurement;

    measurement.z = cv::Mat1d(static_cast<int>(2 * z.size()), 1);

    for (int i = 0; i < measurement.z.rows / 2; ++i) {
        measurement.z(2 * i) = z[static_cast<size_t>(i)].x;
        measurement.z(2 * i + 1) = z[static_cast<size_t>(i)].y;
    }

    measurement.measured = measured;

    return measurement;
}
