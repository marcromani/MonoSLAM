#include "tracking.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

namespace {

// Seeds
constexpr int maxCorners = 150;                 // Generic corners whose motion the seeds follow
constexpr std::size_t minTrackedCorners = 120;  // New ones are only detected when fewer are tracked
constexpr double cornerQuality = 0.01;
constexpr double minCornerDistance = 8.0;  // Pixels
constexpr std::size_t numNeighbors = 24;   // Corners that move a seed
constexpr std::size_t minNeighbors = 6;

constexpr int pyramidLevels = 5;
const cv::Size cornerWindow(21, 21);
const cv::Size featureWindow(31, 31);

constexpr double maxForwardBackwardError = 1.0;  // Pixels
constexpr double motionInlierThreshold = 3.0;    // Pixels
constexpr double maxFlowDisagreement = 6.0;      // Pixels

// Patch search
constexpr int searchRadius = 15;         // Around the seeds, in pixels
constexpr double returningSigmas = 3.0;  // Of the innovation, around the prediction of a returning feature
constexpr int maxReturningRadius = 40;   // Pixels, so that a large uncertainty does not invite look-alikes
constexpr int minPatchPixels = 25;
constexpr double minScore = 0.6;     // Of the correlation
constexpr double minPeakGap = 0.05;  // Between the best peak and any other, of the correlation
constexpr int peakRadius = 2;        // Of the best peak, within which there are no others, in pixels

/*
 * Moves a point according to the corners around it, with the affine motion that most of the
 * nearest ones agree on. Returns the point as it is if there are not enough of them or their
 * motion cannot be explained.
 *
 * point    Point to move
 * from     Corners in the previous frame
 * to       The same corners in the current frame
 */
cv::Point2f
moveWithNeighbors(const cv::Point2f& point, const std::vector<cv::Point2f>& from, const std::vector<cv::Point2f>& to)
{
    if (from.size() < minNeighbors) {
        return point;
    }

    const std::size_t count = std::min(numNeighbors, from.size());

    std::vector<std::size_t> order(from.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });

    std::partial_sort(
      order.begin(),
      order.begin() + static_cast<std::ptrdiff_t>(count),
      order.end(),
      [&](std::size_t a, std::size_t b) { return cv::norm(from[a] - point) < cv::norm(from[b] - point); });

    std::vector<cv::Point2f> neighborsFrom, neighborsTo;

    for (std::size_t i = 0; i < count; ++i) {
        neighborsFrom.push_back(from[order[i]]);
        neighborsTo.push_back(to[order[i]]);
    }

    const cv::Mat motion =
      cv::estimateAffine2D(neighborsFrom, neighborsTo, cv::noArray(), cv::RANSAC, motionInlierThreshold);

    if (motion.empty()) {
        return point;
    }

    const cv::Matx23d A(motion);

    return cv::Point2f(static_cast<float>(A(0, 0) * point.x + A(0, 1) * point.y + A(0, 2)),
                       static_cast<float>(A(1, 0) * point.x + A(1, 1) * point.y + A(1, 2)));
}

/*
 * Returns the corners of a rectangle, clockwise from the top left one.
 */
std::vector<cv::Point2f>
cornersOf(const cv::Rect& rect)
{
    const cv::Point2f topLeft(rect.tl());
    const cv::Point2f bottomRight(rect.br());

    return { topLeft, cv::Point2f(bottomRight.x, topLeft.y), bottomRight, cv::Point2f(topLeft.x, bottomRight.y) };
}

/*
 * Warps the patch of a feature to how it would look from another pose, taking it as a piece of the
 * plane through the feature with the feature normal. A true homography between the views only
 * exists for a pinhole camera, so the corners of the patch are mapped through the whole camera
 * model, and the homography is the one that maps them. Returns the warped patch, the pixels of it
 * that come from the patch, and where the center of the patch lands in it, or an empty patch if
 * the plane is seen from behind or out of the frame.
 *
 * feature     Feature
 * position    Feature position, in world coordinates
 * camera      Camera model
 * pose        Camera pose of the new view
 * anchor      Center of the patch in the warped patch
 * mask        Pixels of the warped patch that come from the patch
 */
cv::Mat1b
warpPatch(const monoslam::Feature& feature,
          const cv::Vec3d& position,
          const monoslam::CameraModel& camera,
          const monoslam::CameraPose& pose,
          cv::Point2d& anchor,
          cv::Mat1b& mask)
{
    const cv::Mat1b& view = feature.frame;
    const std::vector<cv::Point2f> corners = cornersOf(feature.roi);

    // The rays through the corners of the patch in the first view, intersected with the plane and
    // projected into the second one
    const monoslam::CameraPose& first = feature.pose;
    const double m = feature.normal.dot(position - first.r());

    std::vector<cv::Point2f> warpedCorners;

    for (const cv::Point2f& corner : corners) {
        const cv::Vec3d direction = first.R() * camera.unproject(corner);
        const cv::Vec3d point = first.r() + direction * (m / feature.normal.dot(direction));
        const cv::Vec3d pointCamera = pose.Rcw() * (point - pose.r());

        if (pointCamera[2] <= 0.0) {
            return {};
        }

        warpedCorners.emplace_back(camera.distort(pointCamera[0] / pointCamera[2], pointCamera[1] / pointCamera[2]));
    }

    const cv::Matx33d H(cv::getPerspectiveTransform(corners, warpedCorners));

    // Only the box around the warped patch is warped, not the whole view
    std::vector<cv::Point2f> box;
    cv::perspectiveTransform(corners, box, cv::Matx33f(H));

    const cv::Rect patchBox = cv::boundingRect(box) & cv::Rect(0, 0, view.cols, view.rows);

    if (patchBox.empty()) {
        return {};
    }

    const cv::Matx33d toBox(1.0, 0.0, -patchBox.x, 0.0, 1.0, -patchBox.y, 0.0, 0.0, 1.0);

    cv::Mat1b warped;
    cv::warpPerspective(view, warped, toBox * H, patchBox.size());

    cv::Mat1b shape(view.size(), 0);
    shape(feature.roi).setTo(255);
    cv::warpPerspective(shape, mask, toBox * H, patchBox.size());

    const cv::Vec3d center = H * cv::Vec3d(feature.position.x, feature.position.y, 1);

    anchor.x = center[0] / center[2] - patchBox.x;
    anchor.y = center[1] / center[2] - patchBox.y;

    return warped;
}

/*
 * Fits a parabola to the neighbors of a peak and returns the offset of its vertex.
 */
double
refinePeak(double left, double peak, double right)
{
    const double denominator = left - 2.0 * peak + right;

    if (std::fabs(denominator) < 1e-6) {
        return 0.0;
    }

    return std::clamp(0.5 * (left - right) / denominator, -1.0, 1.0);
}

}

monoslam::FlowTracker::FlowTracker(const cv::Mat1b& frame, const std::vector<cv::Point2f>& positions)
  : frame_(frame.clone())
  , seeds_(positions)
{
    detectCorners();
}

std::vector<cv::Point2f>&
monoslam::FlowTracker::predict(const cv::Mat1b& frame)
{
    // The generic corners that the flow follows forwards and backwards alike
    std::vector<cv::Point2f> from, to;

    if (!corners_.empty()) {
        std::vector<cv::Point2f> forward, backward;
        std::vector<uchar> forwardStatus, backwardStatus;
        std::vector<float> error;

        cv::calcOpticalFlowPyrLK(frame_, frame, corners_, forward, forwardStatus, error, cornerWindow, pyramidLevels);
        cv::calcOpticalFlowPyrLK(frame, frame_, forward, backward, backwardStatus, error, cornerWindow, pyramidLevels);

        for (std::size_t i = 0; i < corners_.size(); ++i) {
            if (forwardStatus[i] &&
                backwardStatus[i] &&
                cv::norm(corners_[i] - backward[i]) < maxForwardBackwardError) {
                from.push_back(corners_[i]);
                to.push_back(forward[i]);
            }
        }
    }

    std::vector<cv::Point2f> moved;

    for (const cv::Point2f& seed : seeds_) {
        moved.push_back(moveWithNeighbors(seed, from, to));
    }

    // The flow of each seed itself, which only exists for those in view
    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);

    std::vector<std::size_t> inView;
    std::vector<cv::Point2f> inViewSeeds;

    for (std::size_t i = 0; i < seeds_.size(); ++i) {
        if (frameRect.contains(seeds_[i])) {
            inView.push_back(i);
            inViewSeeds.push_back(seeds_[i]);
        }
    }

    std::vector<cv::Point2f> flow;
    std::vector<uchar> status;
    std::vector<float> error;

    if (!inViewSeeds.empty()) {
        cv::calcOpticalFlowPyrLK(frame_, frame, inViewSeeds, flow, status, error, featureWindow, pyramidLevels);
    }

    for (std::size_t k = 0; k < inView.size(); ++k) {
        const std::size_t i = inView[k];

        if (status[k] && cv::norm(flow[k] - moved[i]) < maxFlowDisagreement) {
            moved[i] = flow[k];
        }
    }

    seeds_ = moved;
    frame_ = frame.clone();

    // The corners followed are kept for the next frame, and replaced only when too many are lost
    corners_ = to;

    if (corners_.size() < minTrackedCorners) {
        detectCorners();
    }

    return seeds_;
}

void
monoslam::FlowTracker::detectCorners()
{
    // Away from the corners still tracked, so that they are not detected twice
    cv::Mat1b mask(frame_.size(), 255);

    for (const cv::Point2f& corner : corners_) {
        cv::circle(mask, corner, static_cast<int>(minCornerDistance), cv::Scalar(0), cv::FILLED);
    }

    std::vector<cv::Point2f> corners;
    const int missing = maxCorners - static_cast<int>(corners_.size());

    if (missing > 0) {
        cv::goodFeaturesToTrack(frame_, corners, missing, cornerQuality, minCornerDistance, mask);
        corners_.insert(corners_.end(), corners.begin(), corners.end());
    }
}

void
monoslam::FlowTracker::remove(const std::vector<std::size_t>& features)
{
    // Backwards, so that the indices still to remove stay valid
    for (auto it = features.rbegin(); it != features.rend(); ++it) {
        seeds_.erase(seeds_.begin() + static_cast<std::ptrdiff_t>(*it));
    }
}

std::optional<cv::Point2d>
monoslam::findPatch(const cv::Mat1b& frame,
                    const Feature& feature,
                    const cv::Vec3d& position,
                    const CameraModel& camera,
                    const CameraPose& pose,
                    const cv::Point2d& center,
                    int radius)
{
    cv::Point2d anchor;
    cv::Mat1b mask;

    const cv::Mat1b patch = warpPatch(feature, position, camera, pose, anchor, mask);

    if (patch.empty() || cv::countNonZero(mask) < minPatchPixels) {
        return std::nullopt;
    }

    // The center of the patch can fall out of it if the frame borders cut the warped patch
    if (anchor.x < 0.0 || anchor.y < 0.0 || anchor.x >= patch.cols || anchor.y >= patch.rows) {
        return std::nullopt;
    }

    const int size = 2 * radius + 1;
    const cv::Rect region = cv::Rect(cvRound(center.x) - radius, cvRound(center.y) - radius, size, size) &
                            cv::Rect(0, 0, frame.cols, frame.rows);

    if (region.empty()) {
        return std::nullopt;
    }

    // The part of the frame that every position of the anchor in the region needs, with black
    // borders where it goes out of the frame, so that the correlation has one entry per position
    const int x0 = region.x - static_cast<int>(std::floor(anchor.x));
    const int y0 = region.y - static_cast<int>(std::floor(anchor.y));
    const int x1 = region.x + region.width + patch.cols - static_cast<int>(std::floor(anchor.x)) - 2;
    const int y1 = region.y + region.height + patch.rows - static_cast<int>(std::floor(anchor.y)) - 2;

    const cv::Rect inFrame =
      cv::Rect(cv::Point(x0, y0), cv::Point(x1 + 1, y1 + 1)) & cv::Rect(0, 0, frame.cols, frame.rows);

    cv::Mat image;
    cv::copyMakeBorder(frame(inFrame),
                       image,
                       inFrame.y - y0,
                       y1 + 1 - (inFrame.y + inFrame.height),
                       inFrame.x - x0,
                       x1 + 1 - (inFrame.x + inFrame.width),
                       cv::BORDER_CONSTANT,
                       cv::Scalar::all(0.0));

    cv::Mat1f result;
    cv::matchTemplate(image, patch, result, cv::TM_CCOEFF_NORMED, mask);
    cv::patchNaNs(result, -1.0);

    // The best peak
    double score;
    cv::Point peak;
    cv::minMaxLoc(result, nullptr, &score, nullptr, &peak);

    if (score < minScore) {
        return std::nullopt;
    }

    // The match is ambiguous if another peak, a local maximum away from the best one, scores almost
    // as high: a repeated texture, or an edge that only fixes one direction. A tiny change of the
    // prediction could then pick either, so neither is used. A local maximum on the border of the
    // region may just be the slope of a peak outside it, and does not count
    cv::Mat1f dilated;
    cv::dilate(result, dilated, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar::all(2.0));

    cv::Mat1f others(result.size(), -1.0f);
    result.copyTo(others, result >= dilated);
    cv::circle(others, peak, peakRadius, cv::Scalar::all(-1.0), cv::FILLED);

    double otherScore;
    cv::minMaxLoc(others, nullptr, &otherScore);

    if (score - otherScore < minPeakGap) {
        return std::nullopt;
    }

    // To subpixel precision
    cv::Point2d found(x0 + anchor.x + peak.x, y0 + anchor.y + peak.y);

    if (0 < peak.x && peak.x < result.cols - 1) {
        found.x += refinePeak(result(peak.y, peak.x - 1), score, result(peak.y, peak.x + 1));
    }

    if (0 < peak.y && peak.y < result.rows - 1) {
        found.y += refinePeak(result(peak.y - 1, peak.x), score, result(peak.y + 1, peak.x));
    }

    return found;
}

std::vector<std::optional<cv::Point2d>>
monoslam::matchFeatures(const cv::Mat1b& frame,
                        const std::vector<Feature>& features,
                        const CameraModel& camera,
                        const EKF& ekf,
                        const Observation& o,
                        const std::vector<cv::Point2f>& seeds)
{
    const cv::Mat1d& x = ekf.state();
    const CameraPose pose = ekf.pose();

    std::vector<std::optional<cv::Point2d>> matches;

    for (std::size_t k = 0; k < o.features.size(); ++k) {
        const std::size_t i = o.features[k];

        const int row = static_cast<int>(2 * k);
        const cv::Point2d prediction(o.h(row), o.h(row + 1));

        // A returning feature has no seed worth the name: the flow could not follow it while it was
        // away. Where it is comes from the filter alone, within the uncertainty of its prediction,
        // which grew with the drift of the camera while it was away
        if (features[i].returning) {
            const double sigma = std::sqrt(std::max(o.S(row, row), o.S(row + 1, row + 1)));
            const int radius =
              std::clamp(static_cast<int>(std::ceil(returningSigmas * sigma)), searchRadius, maxReturningRadius);

            matches.push_back(findPatch(frame, features[i], featurePosition(x, i), camera, pose, prediction, radius));
            continue;
        }

        // Without a seed in view, the prediction is the best guess
        const cv::Point2d center = camera.inView(seeds[i]) ? cv::Point2d(seeds[i]) : prediction;

        matches.push_back(findPatch(frame, features[i], featurePosition(x, i), camera, pose, center, searchRadius));
    }

    return matches;
}
