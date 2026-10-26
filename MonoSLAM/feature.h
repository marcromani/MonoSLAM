#ifndef MONOSLAM_FEATURE_H
#define MONOSLAM_FEATURE_H

#include <opencv2/core.hpp>

namespace monoslam {

class Feature
{
  public:
    cv::Mat1b frame;  // Original grayscale frame
    cv::Rect roi;     // Patch region in the original frame

    cv::Vec3d normal;  // Patch normal vector, in world coordinates

    cv::Matx33d R;  // Camera rotation in the original frame (world to camera)
    cv::Vec3d t;    // Camera location in the original frame, in world coordinates

    int matchingFailures;  // Number of matching failures
    int matchingAttempts;  // Number of matching attempts

    cv::Vec3d initDirection;              // Pre-initialization unit vector from camera to feature, in world coordinates
    cv::Vec<double, 100> initDepths;      // Pre-initialization depth hypotheses
    cv::Vec<double, 100> initDepthProbs;  // Pre-initialization depth probabilities

    bool initialized;  // Whether the feature is fully initialized or not

    Feature() = default;  // Needs to be default-constructible for the monoslam::Buffer

    /*
     * Constructor for a fully initialized feature. It is a convenient method used only
     * during map initialization, when manually initialized features are added with zero
     * position uncertainty.
     *
     * frame     Grayscale camera frame at the time the feature is initialized
     * roi       Feature patch
     * normal    Patch normal vector, in world coordinates
     * R         Camera rotation at feature initialization (world to camera)
     * t         Camera position at feature initialization, in world coordinates
     */
    Feature(const cv::Mat1b& frame,
            const cv::Rect& roi,
            const cv::Vec3d& normal,
            const cv::Matx33d& R,
            const cv::Vec3d& t);

    /*
     * Constructor for a pre-initialized (candidate) feature.
     *
     * frame             Grayscale camera frame at the time the feature is initialized
     * roi               Feature patch
     * R                 Camera rotation at feature initialization (world to camera)
     * t                 Camera position at feature initialization, in world coordinates
     * initDirection     Unit vector from camera to feature, in world coordinates
     * initDepthRange    Interval for the uniformly distributed depth hypotheses
     */
    Feature(const cv::Mat1b& image,
            const cv::Rect& roi,
            const cv::Matx33d& R,
            const cv::Vec3d& t,
            const cv::Vec3d& initDirection,
            const cv::Point2d& initDepthRange);
};

}

#endif
