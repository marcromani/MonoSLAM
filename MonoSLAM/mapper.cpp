#include "mapper.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>

#include <opencv2/features.hpp>
#include <opencv2/imgproc.hpp>

#include "inverse_depth.h"
#include "relocalization.h"

namespace {

constexpr std::size_t minVisible = 12;     // Features are added below this number in view
constexpr std::size_t maxNewPerFrame = 2;  // And at most this many per frame

constexpr int boxDivisions = 4;  // The box where a feature is added is this many times smaller than the frame

constexpr int patchSize = monoslam::Feature::patchSize;
constexpr int border = 10;              // Distance of new features to the border of the frame, in pixels
constexpr int minFeatureDistance = 15;  // Distance of new features to those in view, in pixels

constexpr int maxCandidates = 10;             // In the box
constexpr double candidateQuality = 0.01;     // Relative to the best corner in the box
constexpr double minTexture = 4.0;            // Gray levels per pixel, see texture
constexpr double minCandidateDistance = 5.0;  // Between candidates, in pixels

constexpr int distinctivenessRadius = 20;  // Region where a candidate must not look like anything else
constexpr int selfRadius = 2;              // Around the candidate itself, which is excluded
constexpr double maxLookAlikeScore = 0.8;  // Correlation of the best look-alike allowed

constexpr double defaultInverseDepth = 0.02;  // 1 / cm, that is 50 cm away, without well located features

constexpr int maxDescriptors = 8;    // Per feature
constexpr double minNovelty = 30.0;  // Bits, from the closest descriptor the feature already has

constexpr int minAttempts = 10;     // Before a feature can be removed
constexpr int recentAttempts = 20;  // Only the last ones count, at most this many
constexpr int minFramesAway = 10;   // Out of view, after which the count of a feature starts again
constexpr double minMatchRatio = 0.5;
constexpr int establishedUses = 30;  // Uses after which a feature is never removed

/*
 * Returns the inverse of the median distance of the well located features in view, the best guess for the depth
 * of a new one, or defaultInverseDepth if there are none.
 *
 * ekf       Filter
 * camera    Camera model
 */
double
typicalInverseDepth(const monoslam::EKF& ekf, const monoslam::CameraModel& camera)
{
    const cv::Mat1d& x = ekf.state();

    const monoslam::CameraPose pose = ekf.pose();

    std::vector<double> distances;

    for (std::size_t i = 0; i < monoslam::numFeatures(x); ++i) {
        const cv::Vec3d point = monoslam::featurePosition(x, i);

        if (ekf.isWellLocated(i) && camera.project(pose, point).has_value()) {
            distances.push_back(cv::norm(point - pose.r()));
        }
    }

    if (distances.empty()) {
        return defaultInverseDepth;
    }

    std::nth_element(
      distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2), distances.end());

    return 1.0 / distances[distances.size() / 2];
}

/*
 * Slides a box a fraction of the size of the frame over it, by half its size, and returns the positions with no feature
 * inside. The positions farthest from the features come first, since that is where a new one is most needed.
 * Davison et al. place the box randomly instead.
 *
 * inside      Region of the frame where features can be added
 * features    Image positions of the features in view
 */
std::vector<cv::Rect>
findEmptyBoxes(const cv::Rect& inside, const std::vector<cv::Point2f>& features)
{
    const int width = inside.width / boxDivisions;
    const int height = inside.height / boxDivisions;

    std::vector<std::pair<double, cv::Rect>> boxes;  // And how far they are from the features

    for (int y = inside.y; y + height <= inside.y + inside.height; y += height / 2) {
        for (int x = inside.x; x + width <= inside.x + inside.width; x += width / 2) {
            const cv::Rect box(x, y, width, height);

            const bool empty = std::none_of(
              features.begin(), features.end(), [&](const cv::Point2f& feature) { return box.contains(feature); });

            if (!empty) {
                continue;
            }

            // Without features in view, the box closest to the center of the frame
            const cv::Point2f center(static_cast<float>(x) + 0.5f * static_cast<float>(width),
                                     static_cast<float>(y) + 0.5f * static_cast<float>(height));

            double distance =
              -cv::norm(center - cv::Point2f(static_cast<float>(inside.x) + 0.5f * static_cast<float>(inside.width),
                                             static_cast<float>(inside.y) + 0.5f * static_cast<float>(inside.height)));

            if (!features.empty()) {
                distance = std::numeric_limits<double>::max();

                for (const cv::Point2f& feature : features) {
                    distance = std::min(distance, static_cast<double>(cv::norm(center - feature)));
                }
            }

            boxes.emplace_back(distance, box);
        }
    }

    std::stable_sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<cv::Rect> sorted;

    for (const auto& [distance, box] : boxes) {
        sorted.push_back(box);
    }

    return sorted;
}

/*
 * Returns how well the patch around a pixel can be located along its least textured direction: the root mean square
 * of the intensity gradient along that direction, in gray levels per pixel. This is the smaller eigenvalue
 * of the gradient matrix of the patch, as in Shi and Tomasi (1994). Here it is absolute rather than relative
 * to the best corner around, so that a blank wall scores low however blank the rest of the frame is. The correlation
 * of the patch search cannot tell this: it divides by the contrast of the patch, and on a blank wall what is left
 * is camera noise.
 *
 * frame    Frame
 * pixel    Center of the patch
 */
double
texture(const cv::Mat1b& frame, const cv::Point2f& pixel)
{
    // One more pixel on every side, for the derivatives on the border of the patch
    const cv::Rect around(
      cvRound(pixel.x) - patchSize / 2 - 1, cvRound(pixel.y) - patchSize / 2 - 1, patchSize + 2, patchSize + 2);
    const cv::Rect patch(1, 1, patchSize, patchSize);

    // The Sobel kernel is 8 times the derivative
    cv::Mat1f gx, gy;
    cv::Sobel(frame(around), gx, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(frame(around), gy, CV_32F, 0, 1, 3, 1.0 / 8.0);

    const double xx = gx(patch).dot(gx(patch));
    const double yy = gy(patch).dot(gy(patch));
    const double xy = gx(patch).dot(gy(patch));

    const double smallest = 0.5 * (xx + yy - std::sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy));

    return std::sqrt(std::max(smallest, 0.0) / (patchSize * patchSize));
}

/*
 * Returns whether the patch around a pixel does not look like any other place near it.
 *
 * frame    Frame
 * pixel    Center of the patch
 */
bool
isDistinctive(const cv::Mat1b& frame, const cv::Point2f& pixel)
{
    const cv::Rect patch(cvRound(pixel.x) - patchSize / 2, cvRound(pixel.y) - patchSize / 2, patchSize, patchSize);

    const int regionSize = patchSize + 2 * distinctivenessRadius;
    const cv::Rect region =
      cv::Rect(patch.x - distinctivenessRadius, patch.y - distinctivenessRadius, regionSize, regionSize) &
      cv::Rect(0, 0, frame.cols, frame.rows);

    cv::Mat1f result;
    cv::matchTemplate(frame(region), frame(patch), result, cv::TM_CCOEFF_NORMED);
    cv::patchNaNs(result, -1.0);

    // The patch matches itself perfectly, wherever it is in the region
    const cv::Point self(patch.x - region.x, patch.y - region.y);
    const cv::Rect around(self.x - selfRadius, self.y - selfRadius, 2 * selfRadius + 1, 2 * selfRadius + 1);

    result(around & cv::Rect(0, 0, result.cols, result.rows)).setTo(-1.0f);

    double lookAlike;
    cv::minMaxLoc(result, 0, &lookAlike);

    return lookAlike < maxLookAlikeScore;
}

}

monoslam::Mapper::Mapper(std::size_t firstId)
  : nextId_(firstId)
{
}

void
monoslam::Mapper::recordMatches(std::vector<Feature>& features,
                                const std::vector<std::size_t>& searched,
                                const std::vector<std::size_t>& used) const
{
    for (std::size_t i = 0; i < features.size(); ++i) {
        Feature& feature = features[i];

        if (std::find(searched.begin(), searched.end(), i) == searched.end()) {
            feature.returning = ++feature.framesAway >= minFramesAway || feature.returning;
            continue;
        }

        // A feature that comes back into view after a while starts counting again. Its failures before it left, near
        // the border or blurred by the turn, are not the feature's fault, and neither is the drift of the pose while it
        // was away
        if (feature.framesAway >= minFramesAway) {
            feature.matchingAttempts = 0;
            feature.matchHistory = 0;
        }

        const bool matched = std::find(used.begin(), used.end(), i) != used.end();

        feature.timesUsed += matched ? 1 : 0;
        feature.framesAway = 0;
        feature.returning = feature.returning && !matched;
        ++feature.matchingAttempts;
        feature.matchHistory = (feature.matchHistory << 1) | (matched ? 1u : 0u);
    }
}

void
monoslam::Mapper::recordAppearance(const cv::Mat1b& frame,
                                   std::vector<Feature>& features,
                                   const std::vector<std::size_t>& used,
                                   const std::vector<cv::Point2f>& positions) const
{
    for (std::size_t i : used) {
        Feature& feature = features[i];

        if (feature.descriptors.rows >= maxDescriptors) {
            continue;
        }

        const cv::Mat1b descriptor = describe(frame, cv::Point2d(positions[i]));

        if (descriptor.empty()) {
            continue;
        }

        double closest = std::numeric_limits<double>::max();

        for (int k = 0; k < feature.descriptors.rows; ++k) {
            closest = std::min(closest, cv::norm(descriptor, feature.descriptors.row(k), cv::NORM_HAMMING));
        }

        if (closest > minNovelty) {
            feature.descriptors.push_back(descriptor);
        }
    }
}

void
monoslam::Mapper::removeUnreliable(EKF& ekf, std::vector<Feature>& features, FlowTracker& tracker) const
{
    std::vector<std::size_t> unreliable;

    for (std::size_t i = 0; i < features.size(); ++i) {
        const Feature& feature = features[i];

        // The corners of the pattern hold the scale of the map, and a feature used for long is part of the map
        // that the camera can come back to and close the loop with. Either may fail for a while, blurred, occluded
        // or seen from far away, without being a bad feature
        if (feature.knownSurface || feature.timesUsed >= establishedUses) {
            continue;
        }

        if (feature.matchingAttempts < minAttempts) {
            continue;
        }

        // Only the recent attempts count: a feature that was reliable for a long time and is now occluded, or was
        // on something that moved, must be removed as quickly as a new one
        const int recent = std::min(feature.matchingAttempts, recentAttempts);
        const std::uint32_t mask = (1u << recent) - 1u;
        const auto matches = static_cast<double>(std::bitset<32>(feature.matchHistory & mask).count());

        if (matches < minMatchRatio * recent) {
            unreliable.push_back(i);
        }
    }

    if (unreliable.empty()) {
        return;
    }

    ekf.removeFeatures(unreliable);

    tracker.remove(unreliable);

    for (auto it = unreliable.rbegin(); it != unreliable.rend(); ++it) {
        features.erase(features.begin() + static_cast<std::ptrdiff_t>(*it));
    }
}

void
monoslam::Mapper::addFeatures(const cv::Mat1b& frame,
                              const CameraModel& camera,
                              EKF& ekf,
                              std::vector<Feature>& features,
                              FlowTracker& tracker)
{
    const cv::Rect inside(border, border, frame.cols - 2 * border, frame.rows - 2 * border);

    // The features in view, where new ones must not be
    std::vector<cv::Point2f> inView;

    const std::vector<cv::Point2f> projections =
      ekf.project(std::vector<cv::Point2f>(features.size(), cv::Point2f(-1.0f, -1.0f)));

    for (const cv::Point2f& projection : projections) {
        if (camera.inView(projection)) {
            inView.push_back(projection);
        }
    }

    if (inView.size() >= minVisible) {
        return;
    }

    const CameraPose pose = ekf.pose();

    // New features are probably about as far as the others, but they may be anywhere from a third of that distance
    // to infinity, within two standard deviations
    const double inverseDepth = typicalInverseDepth(ekf, camera);
    const double inverseDepthStd = inverseDepth;

    std::size_t added = 0;

    // The boxes overlap, and are searched again for every feature added, so the same corner can be a candidate many
    // times: whether it is good enough is only computed once
    std::map<std::pair<float, float>, bool> isGood;

    const auto good = [&](const cv::Point2f& pixel) {
        const auto [known, inserted] = isGood.try_emplace({ pixel.x, pixel.y }, false);

        if (inserted) {
            known->second = texture(frame, pixel) >= minTexture && isDistinctive(frame, pixel);
        }

        return known->second;
    };

    while (inView.size() < minVisible && added < maxNewPerFrame) {
        // The most salient distinctive corner of the emptiest box that has one, away from the features near it. A box
        // can be empty because there is nothing to track there
        std::optional<cv::Point2f> found;

        for (const cv::Rect& box : findEmptyBoxes(inside, inView)) {
            // The corners are only looked for in the box, with a margin around it so that those on its border
            // are scored with their true neighbors
            const cv::Rect region = (box + cv::Size(8, 8) - cv::Point(4, 4)) & cv::Rect(0, 0, frame.cols, frame.rows);

            cv::Mat1b mask(region.size(), 0);
            mask(box - region.tl()).setTo(255);

            for (const cv::Point2f& feature : inView) {
                cv::circle(mask, cv::Point(feature) - region.tl(), minFeatureDistance, cv::Scalar(0), cv::FILLED);
            }

            std::vector<cv::Point2f> candidates;
            cv::goodFeaturesToTrack(
              frame(region), candidates, maxCandidates, candidateQuality, minCandidateDistance, mask);

            for (cv::Point2f& candidate : candidates) {
                candidate += cv::Point2f(region.tl());
            }

            const auto candidate = std::find_if(candidates.begin(), candidates.end(), good);

            if (candidate != candidates.end()) {
                found = *candidate;
                break;
            }
        }

        if (!found.has_value()) {
            break;
        }

        ekf.addFeature(found.value(), inverseDepth, inverseDepthStd);

        // The patch is taken as facing the camera, since nothing is known about its surface
        const cv::Mat1d& y = ekf.state();
        const cv::Vec3d normal = -inverse_depth::direction(y(y.rows - 3), y(y.rows - 2));

        features.emplace_back(frame, found.value(), normal, pose, nextId_++);
        tracker.add(found.value());
        inView.push_back(found.value());

        ++added;
    }
}
