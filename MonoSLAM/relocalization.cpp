#include "relocalization.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>

#include "tracking.h"

namespace {

// Descriptors
constexpr int patchSize = 31;  // Of ORB, in pixels
constexpr int radius = patchSize / 2;

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

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= radius * radius) {
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
constexpr double minCornerDistance = 7.0;  // Pixels

constexpr float maxRatio = 0.8f;     // Of the distances to the best corner and the best one elsewhere
constexpr int numNearest = 8;        // Corners compared with the best one
constexpr double sameCorner = 10.0;  // Corners closer than this to the best one are not elsewhere, in pixels
constexpr float maxDistance = 64;    // Bits, out of 256

constexpr int iterations = 1000;             // Of RANSAC
constexpr float maxReprojectionError = 4.0;  // Pixels
constexpr std::size_t minInliers = 6;        // And patches found where the pose projects them

constexpr int verifyRadius = 8;          // Of the search of the patches, in pixels
constexpr double verifyTolerance = 3.0;  // Pixels

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

}

cv::Mat1b
monoslam::describe(const cv::Mat1b& frame, std::vector<cv::Point2f>& positions)
{
    static const cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // ORB needs its patch, rotated, inside the frame
    const cv::Rect inside(patchSize, patchSize, frame.cols - 2 * patchSize, frame.rows - 2 * patchSize);

    std::vector<cv::KeyPoint> keypoints;

    for (const cv::Point2f& position : positions) {
        const cv::Point center(cvRound(position.x), cvRound(position.y));

        if (inside.contains(center)) {
            keypoints.emplace_back(cv::Point2f(center), patchSize, orientation(frame, center));
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

std::optional<monoslam::GlobalMatch>
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
    std::vector<cv::Point3d> points;
    std::vector<cv::Point2d> pixels;

    for (const auto& [i, best] : bestOfFeature) {
        matched.push_back(i);
        points.emplace_back(featurePosition(x, i));
        pixels.emplace_back(corners[static_cast<std::size_t>(best.trainIdx)]);
    }

    if (matched.size() < minInliers) {
        return std::nullopt;
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
    if (countPatchesAt(frame, features, camera, x, candidates, CameraPose(r, R)) < minInliers) {
        return std::nullopt;
    }

    GlobalMatch match;
    match.r = r;
    match.R = R;

    for (int inlier : inliers) {
        match.features.push_back(matched[static_cast<std::size_t>(inlier)]);
        match.pixels.push_back(pixels[static_cast<std::size_t>(inlier)]);
    }

    return match;
}
