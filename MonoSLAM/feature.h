#ifndef MONOSLAM_FEATURE_H
#define MONOSLAM_FEATURE_H

#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

#include "camera.h"

namespace monoslam {

class Feature
{
  public:
    cv::Mat1b frame;       // Original grayscale frame
    cv::Point2d position;  // Pixel position in the original frame
    cv::Rect roi;          // Patch region in the original frame

    cv::Mat1b descriptors;  // Descriptors of the feature as it has looked so far, one per row

    cv::Vec3d normal;           // Patch normal vector, in world coordinates
    bool knownSurface = false;  // Whether the normal is known, rather than assumed to face the camera

    CameraPose pose;  // Camera pose in the original frame

    std::size_t id = 0;  // Unique identifier, which does not change when other features are removed

    int matchingAttempts = 0;        // Number of times the feature was searched
    std::uint32_t matchHistory = 0;  // Whether the filter used it in each of its last attempts, the last in bit 0

    /*
     * Constructor of a feature, which keeps the frame where it was first seen to take its patch
     * from it.
     *
     * image          Grayscale camera frame at the time the feature is initialized
     * pixel          Feature pixel position
     * patchSize      Feature patch size
     * patchNormal    Patch normal vector, in world coordinates
     * firstPose      Camera pose at feature initialization
     */
    Feature(const cv::Mat1b& image,
            const cv::Point2d& pixel,
            int patchSize,
            const cv::Vec3d& patchNormal,
            const CameraPose& firstPose);
};

}

#endif
