#ifndef FEATURE_H
#define FEATURE_H

#include <vector>

#include "opencv2/core/core.hpp"

class Feature {

public:

    cv::Mat image;                  // Original frame
    cv::Rect roi;                   // Patch region in the original frame
    cv::Mat pos;                    // Patch center, in world coordinates
    cv::Mat normal;                 // Patch normal vector, in world coordinates
    cv::Mat R;                      // Camera rotation in the original frame (world to camera)
    cv::Mat t;                      // Camera location in the original frame, in world coordinates

    cv::Mat P;                      // Feature state (position) covariance matrix

    double failRatio;               // Proportion of times the feature could not be matched

    cv::Point2i currentPos2D;       // Current pixel position (for a visible feature)

    cv::Mat dir;                    // Feature line unit direction vector, in world coordinates
    std::vector<double> depths;     // Feature depth hypotheses
    std::vector<double> probs;      // Hypotheses probabilities

    /*
     * Constructor for a fully initialized feature.
     */
    Feature(const cv::Mat& image_, const cv::Rect& roi_, const cv::Mat& normal_,
            const cv::Mat& R_, const cv::Mat& t_);

    /*
     * Constructor for a pre-initialized feature.
     *
     * depthInterval    Interval which contains the depth hypotheses
     * depthSamples     Number of depth hypotheses
     */
    Feature(const cv::Mat& image_, const cv::Rect& roi_, const cv::Mat& R_, const cv::Mat& t_,
            const cv::Mat& dir_, const cv::Point2d& depthInterval, int depthSamples);
};

#endif