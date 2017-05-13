#ifndef FEATURE_H
#define FEATURE_H

#include <vector>

#include "opencv2/core/core.hpp"

class Feature {

public:

    cv::Mat image;                  // Original grayscale frame
    cv::Rect roi;                   // Patch region in the original frame
    cv::Mat pos;                    // Patch center, in world coordinates
    cv::Mat normal;                 // Patch normal vector, in world coordinates
    cv::Mat R;                      // Camera rotation in the original frame (world to camera)
    cv::Mat t;                      // Camera location in the original frame, in world coordinates

    cv::Mat P;                      // Feature state (position) covariance matrix

    int matchingFails;              // Number of times the feature could not be matched
    int matchingAttempts;           // Number of matching attempts

    cv::Mat dir;                    // Feature line unit direction vector, in world coordinates
    std::vector<double> depths;     // Feature depth hypotheses
    cv::Mat probs;                  // Hypotheses probabilities

    /*
     * Constructor for a fully initialized feature. It is a convenient method used only
     * during the map initialization step, when manually initialized features are added
     * with zero position uncertainty.
     *
     * image    Grayscale camera frame at the time the feature is initialized
     * roi      Feature patch
     * normal   Patch normal vector
     * R        Camera rotation (world to camera) at feature initialization
     * t        Camera position, in world coordinates, at feature initialization
     */
    Feature(const cv::Mat& image, const cv::Rect& roi, const cv::Mat& normal,
            const cv::Mat& R, const cv::Mat& t);

    /*
     * Constructor for a pre-initialized (candidate) feature.
     *
     * image            Grayscale camera frame at the time the feature is initialized
     * roi              Feature patch
     * R                Camera rotation (world to camera) at feature initialization
     * t                Camera position, in world coordinates, at feature initialization
     * dir              Feature line unit direction vector, in world coordinates
     * depthInterval    Interval which contains the depth hypotheses
     * depthSamples     Number of depth hypotheses
     */
    Feature(const cv::Mat& image, const cv::Rect& roi, const cv::Mat& R, const cv::Mat& t,
            const cv::Mat& dir, const cv::Point2d& depthInterval, int depthSamples);
};

#endif