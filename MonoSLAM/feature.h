#ifndef MONOSLAM_FEATURE_H
#define MONOSLAM_FEATURE_H

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

#include "camera.h"

namespace monoslam {

/*
 * A feature of the map as the images see it: the patch it is searched with, taken from the frame
 * where it was first seen, the descriptors it is found with anywhere in a frame, and the record of
 * its matches that the mapper judges it by. Its position is in the state of the filter, at the same
 * index as in the list of features.
 */
class Feature
{
  public:
    static constexpr int patchSize = 11;  // Side of the patch, in pixels

    cv::Mat1b frame;       // Frame where it was first seen, shared
    cv::Point2d position;  // Pixel in that frame
    cv::Rect roi;          // Patch in that frame

    cv::Mat1b descriptors;  // Descriptors of the feature as it has looked so far, one per row

    cv::Vec3d normal;           // Of the patch, in world coordinates
    bool knownSurface = false;  // Whether the normal is known, rather than assumed to face the camera

    CameraPose pose;  // Camera pose in that frame

    std::size_t id;  // Unique identifier, which does not change when other features are removed

    int matchingAttempts = 0;        // Number of times the feature was searched
    std::uint32_t matchHistory = 0;  // Whether the filter used it in each of its last attempts, the last in bit 0
    int framesAway = 0;              // Frames since it was last searched
    bool returning = false;          // Back in view after a while, and not used by the filter since
    int timesUsed = 0;               // Frames in which the filter used it, over its whole life

    /*
     * image          Frame where the feature is first seen
     * pixel          Its pixel
     * patchNormal    Normal of the patch, in world coordinates
     * firstPose      Camera pose
     * uniqueId       Unique identifier
     */
    Feature(const cv::Mat1b& image,
            const cv::Point2d& pixel,
            const cv::Vec3d& patchNormal,
            const CameraPose& firstPose,
            std::size_t uniqueId);
};

}

#endif
