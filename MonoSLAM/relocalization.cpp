#include "relocalization.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include "tracking.h"

namespace {

// Descriptors
constexpr int orbPatchSize = 31;  // Pixels
constexpr int orbRadius = orbPatchSize / 2;

/*
 * Returns the orientation of the patch around a pixel, in degrees: the direction from the pixel to
 * the centroid of the intensities of the disc around it, which turns with the image. ORB only
 * computes it for the keypoints it detects itself.
 */
float
orientation(const cv::Mat1b& frame, const cv::Point& center)
{
    double m10 = 0.0;
    double m01 = 0.0;

    for (int dy = -orbRadius; dy <= orbRadius; ++dy) {
        for (int dx = -orbRadius; dx <= orbRadius; ++dx) {
            if (dx * dx + dy * dy <= orbRadius * orbRadius) {
                const double intensity = frame(center.y + dy, center.x + dx);

                m10 += dx * intensity;
                m01 += dy * intensity;
            }
        }
    }

    const double angle = std::atan2(m01, m10) * 180.0 / CV_PI;

    return static_cast<float>(angle < 0.0 ? angle + 360.0 : angle);
}

// Global matching
constexpr int numCorners = 1500;  // Detected in the frame, as the features are when they are added
constexpr double cornerQuality = 0.002;
constexpr double minCornerDistance = 3.5;  // Pixels

constexpr float maxRatio = 0.8f;    // Of the distances to the best corner and the best one elsewhere
constexpr int numNearest = 8;       // Corners compared with the best one
constexpr double sameCorner = 5.0;  // Corners closer than this to the best one are not elsewhere, in pixels
constexpr float maxDistance = 64;   // Bits, out of 256

constexpr int iterations = 1000;             // Of RANSAC
constexpr float maxReprojectionError = 2.0;  // Pixels
constexpr std::size_t minInliers = 6;        // And patches found where the pose projects them

// Keyframes
const cv::Size thumbnailSize(80, 60);                  // PTAM's are 40 x 30, of 640 x 480 frames
constexpr double thumbnailBlur = 2.5;                  // Pixels of the thumbnail
constexpr int maxThumbnailShift = 6;                   // Pixels of the thumbnail
constexpr std::size_t numClosest = 3;                  // Keyframes tried
constexpr double keyframeDistance = 5.0;               // cm, from every other keyframe, or
constexpr double keyframeAngle = 8.0 * CV_PI / 180.0;  // rad
constexpr std::size_t minKeyframeFeatures = 8;         // Used in the frame
constexpr int keyframeSearchRadius = 10;               // Pixels

constexpr int verifyRadius = 4;          // Of the search of the patches, in pixels
constexpr double verifyTolerance = 1.5;  // Pixels

/*
 * Returns how many features are found by their patches where a pose projects them.
 *
 * frame         Current frame
 * features      All the features
 * camera        Camera model
 * x             State
 * candidates    Features to look for
 * pose          Pose to verify
 */
std::size_t
countPatchesAt(const cv::Mat1b& frame,
               const std::vector<monoslam::Feature>& features,
               const monoslam::CameraModel& camera,
               const cv::Mat1d& x,
               const std::vector<std::size_t>& candidates,
               const monoslam::CameraPose& pose)
{
    std::size_t count = 0;

    for (std::size_t i : candidates) {
        const cv::Vec3d position = monoslam::featurePosition(x, i);
        const std::optional<cv::Point2d> projection = camera.project(pose, position);

        if (!projection.has_value()) {
            continue;
        }

        const std::optional<cv::Point2d> found =
          monoslam::findPatch(frame, features[i], position, camera, pose, projection.value(), verifyRadius);

        if (found.has_value() && cv::norm(found.value() - projection.value()) < verifyTolerance) {
            ++count;
        }
    }

    return count;
}

/*
 * Fits a pose to features found in a frame with P3P and RANSAC, taking their positions as exact,
 * and accepts it if the patches of the candidates are also found where it projects them.
 *
 * frame         Current frame
 * features      All the features
 * camera        Camera model
 * x             State
 * candidates    Features that could be in view
 * matched       Features found
 * pixels        Where they were found
 */
std::optional<monoslam::Relocalization>
fitPose(const cv::Mat1b& frame,
        const std::vector<monoslam::Feature>& features,
        const monoslam::CameraModel& camera,
        const cv::Mat1d& x,
        const std::vector<std::size_t>& candidates,
        const std::vector<std::size_t>& matched,
        const std::vector<cv::Point2d>& pixels)
{
    if (matched.size() < minInliers) {
        return std::nullopt;
    }

    std::vector<cv::Point3d> points;

    for (std::size_t i : matched) {
        points.emplace_back(monoslam::featurePosition(x, i));
    }

    cv::Vec3d rotation;     // From world to camera
    cv::Vec3d translation;  // From world to camera
    std::vector<int> inliers;

    const bool found = cv::solvePnPRansac(points,
                                          pixels,
                                          camera.K(),
                                          camera.distCoeffs(),
                                          rotation,
                                          translation,
                                          false,
                                          iterations,
                                          maxReprojectionError,
                                          0.99,
                                          inliers,
                                          cv::SOLVEPNP_AP3P);

    if (!found || inliers.size() < minInliers) {
        return std::nullopt;
    }

    // solvePnP gives the rotation from world to camera
    const cv::Matx33d R = monoslam::rotation(rotation).t();
    const cv::Vec3d r = -(R * translation);

    // A wrong pose can agree with a few descriptors by chance, but hardly with the patches too
    if (countPatchesAt(frame, features, camera, x, candidates, monoslam::CameraPose(r, R)) < minInliers) {
        return std::nullopt;
    }

    monoslam::Relocalization relocalization;
    relocalization.r = r;
    relocalization.R = R;

    for (int inlier : inliers) {
        relocalization.features.push_back(matched[static_cast<std::size_t>(inlier)]);
        relocalization.pixels.push_back(pixels[static_cast<std::size_t>(inlier)]);
    }

    return relocalization;
}

/*
 * Returns a frame shrunk and blurred, with zero mean, as in Klein and Murray (2008): small enough to
 * compare with many others, and blurred enough to compare across small changes of the pose.
 */
cv::Mat1f
thumbnail(const cv::Mat1b& frame)
{
    cv::Mat1f small;
    cv::resize(cv::Mat1f(frame), small, thumbnailSize, 0.0, 0.0, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(), thumbnailBlur);
    small -= cv::mean(small)[0];

    return small;
}

/*
 * Returns the mean squared difference of two thumbnails, the second shifted by some pixels, over
 * the pixels they share.
 */
double
difference(const cv::Mat1f& a, const cv::Mat1f& b, int dx, int dy)
{
    const cv::Rect shared = cv::Rect(0, 0, a.cols, a.rows) & cv::Rect(dx, dy, a.cols, a.rows);
    cv::Mat1f diff;
    cv::subtract(a(shared), b(shared - cv::Point(dx, dy)), diff);

    return diff.dot(diff) / shared.area();
}

}

void
monoslam::Keyframes::add(const cv::Mat1b& frame, const CameraPose& pose, std::size_t numUsed)
{
    if (numUsed < minKeyframeFeatures) {
        return;
    }

    for (const Keyframe& keyframe : keyframes_) {
        const cv::Matx33d turn = keyframe.pose.R().t() * pose.R();
        const double angle = std::acos(std::clamp((cv::trace(turn) - 1.0) / 2.0, -1.0, 1.0));

        if (cv::norm(keyframe.pose.r() - pose.r()) < keyframeDistance && angle < keyframeAngle) {
            return;
        }
    }

    keyframes_.push_back({ thumbnail(frame), pose });
}

std::optional<monoslam::Relocalization>
monoslam::Keyframes::match(const cv::Mat1b& frame,
                           const std::vector<Feature>& features,
                           const CameraModel& camera,
                           const EKF& ekf) const
{
    const cv::Mat1d& x = ekf.state();
    const cv::Mat1f small = thumbnail(frame);

    // The keyframes that look most like the frame
    std::vector<std::pair<double, std::size_t>> closest;

    for (std::size_t k = 0; k < keyframes_.size(); ++k) {
        closest.emplace_back(difference(small, keyframes_[k].thumbnail, 0, 0), k);
    }

    std::sort(closest.begin(), closest.end());
    closest.resize(std::min(closest.size(), numClosest));

    std::vector<std::size_t> candidates;

    for (std::size_t i = 0; i < features.size(); ++i) {
        if (!features[i].knownSurface && ekf.isWellLocated(i)) {
            candidates.push_back(i);
        }
    }

    const cv::Rect inside(0, 0, frame.cols, frame.rows);

    for (const auto& keyframeDifference : closest) {
        const Keyframe& keyframe = keyframes_[keyframeDifference.second];

        // A small turn of the camera shifts the whole image: the shift that best aligns them
        cv::Point shift;
        double best = difference(small, keyframe.thumbnail, 0, 0);

        for (int dy = -maxThumbnailShift; dy <= maxThumbnailShift; ++dy) {
            for (int dx = -maxThumbnailShift; dx <= maxThumbnailShift; ++dx) {
                const double d = difference(small, keyframe.thumbnail, dx, dy);

                if (d < best) {
                    best = d;
                    shift = cv::Point(dx, dy);
                }
            }
        }

        const cv::Point2d offset(shift.x * frame.cols / double(thumbnailSize.width),
                                 shift.y * frame.rows / double(thumbnailSize.height));

        // The features are searched where the keyframe would see them, shifted
        std::vector<std::size_t> matched;
        std::vector<cv::Point2d> pixels;

        for (std::size_t i : candidates) {
            const cv::Vec3d position = featurePosition(x, i);
            const std::optional<cv::Point2d> projection = camera.project(keyframe.pose, position);

            if (!projection.has_value() || !inside.contains(projection.value() + offset)) {
                continue;
            }

            const std::optional<cv::Point2d> found = findPatch(
              frame, features[i], position, camera, keyframe.pose, projection.value() + offset, keyframeSearchRadius);

            if (found.has_value()) {
                matched.push_back(i);
                pixels.push_back(found.value());
            }
        }

        if (std::optional<Relocalization> found = fitPose(frame, features, camera, x, candidates, matched, pixels)) {
            return found;
        }
    }

    return std::nullopt;
}

cv::Mat1b
monoslam::describe(const cv::Mat1b& frame, std::vector<cv::Point2f>& positions)
{
    static const cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // ORB needs its patch, rotated, inside the frame
    const cv::Rect inside(orbPatchSize, orbPatchSize, frame.cols - 2 * orbPatchSize, frame.rows - 2 * orbPatchSize);

    std::vector<cv::KeyPoint> keypoints;

    for (const cv::Point2f& position : positions) {
        const cv::Point center(cvRound(position.x), cvRound(position.y));

        if (inside.contains(center)) {
            keypoints.emplace_back(cv::Point2f(center), orbPatchSize, orientation(frame, center));
        }
    }

    cv::Mat descriptors;

    if (!keypoints.empty()) {
        orb->compute(frame, keypoints, descriptors);
    }

    positions.clear();

    for (const cv::KeyPoint& keypoint : keypoints) {
        positions.push_back(keypoint.pt);
    }

    return cv::Mat1b(descriptors);
}

cv::Mat1b
monoslam::describe(const cv::Mat1b& frame, const cv::Point2d& position)
{
    std::vector<cv::Point2f> positions = { cv::Point2f(position) };

    return describe(frame, positions);
}

std::optional<monoslam::Relocalization>
monoslam::matchGlobally(const cv::Mat1b& frame,
                        const std::vector<Feature>& features,
                        const CameraModel& camera,
                        const EKF& ekf)
{
    const cv::Mat1d& x = ekf.state();

    // Only the features whose position is known can tell the pose, and those of the pattern are all
    // alike. Each feature has a descriptor for every way it has looked
    std::vector<std::size_t> candidates;
    std::vector<std::size_t> owners;  // Feature of each descriptor
    cv::Mat descriptors;

    for (std::size_t i = 0; i < features.size(); ++i) {
        if (!features[i].knownSurface && !features[i].descriptors.empty() && ekf.isWellLocated(i)) {
            candidates.push_back(i);
            owners.insert(owners.end(), static_cast<std::size_t>(features[i].descriptors.rows), i);
            descriptors.push_back(features[i].descriptors);
        }
    }

    if (candidates.size() < minInliers) {
        return std::nullopt;
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame, corners, numCorners, cornerQuality, minCornerDistance);

    const cv::Mat1b frameDescriptors = describe(frame, corners);

    if (corners.size() < 2) {
        return std::nullopt;
    }

    // Each feature is matched to its nearest corner, if it is clearly nearer than any corner
    // elsewhere. The corners around the same structure look alike, but are not an ambiguity
    std::vector<std::vector<cv::DMatch>> nearest;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(descriptors, frameDescriptors, nearest, numNearest);

    // The best match of each feature, over all its descriptors
    std::map<std::size_t, cv::DMatch> bestOfFeature;

    for (const std::vector<cv::DMatch>& candidatesOfDescriptor : nearest) {
        if (candidatesOfDescriptor.empty() || candidatesOfDescriptor[0].distance >= maxDistance) {
            continue;
        }

        const cv::DMatch& best = candidatesOfDescriptor[0];
        const cv::Point2f& corner = corners[static_cast<std::size_t>(best.trainIdx)];

        const auto elsewhere =
          std::find_if(candidatesOfDescriptor.begin() + 1, candidatesOfDescriptor.end(), [&](const cv::DMatch& other) {
              return cv::norm(corners[static_cast<std::size_t>(other.trainIdx)] - corner) > sameCorner;
          });

        if (elsewhere != candidatesOfDescriptor.end() && best.distance >= maxRatio * elsewhere->distance) {
            continue;
        }

        const std::size_t i = owners[static_cast<std::size_t>(best.queryIdx)];
        const auto found = bestOfFeature.find(i);

        if (found == bestOfFeature.end() || best.distance < found->second.distance) {
            bestOfFeature[i] = best;
        }
    }

    std::vector<std::size_t> matched;
    std::vector<cv::Point2d> pixels;

    for (const auto& [i, best] : bestOfFeature) {
        matched.push_back(i);
        pixels.emplace_back(corners[static_cast<std::size_t>(best.trainIdx)]);
    }

    return fitPose(frame, features, camera, x, candidates, matched, pixels);
}
