#include "feature.hpp"

using namespace cv;

/*
 * Constructor for a fully initialized feature.
 */
Feature::Feature(const Mat& image_, const Rect& roi_, const Mat& normal_,
                 const Mat& R_, const Mat& t_) {

    image_.copyTo(image);
    roi = roi_;
    normal_.copyTo(normal);
    R_.copyTo(R);
    t_.copyTo(t);

    failRatio = 0;
}

/*
 * Constructor for a pre-initialized feature.
 *
 * depthInterval    Interval which contains the depth hypotheses
 * depthSamples     Number of depth hypotheses
 */
Feature::Feature(const Mat& image_, const Rect& roi_, const Mat& R_, const Mat& t_,
                 const Mat& dir_, const Point2d& depthInterval, int depthSamples) {

    image_.copyTo(image);
    roi = roi_;
    R_.copyTo(R);
    t_.copyTo(t);
    dir_.copyTo(dir);

    depths.resize(depthSamples);

    double step = (depthInterval.y - depthInterval.x) / (depthSamples - 1);

    for (int i = 0; i < depthSamples; i++)
        depths[i] = depthInterval.x + step * i;

    probs.resize(depthSamples, 1.0 / depthSamples);

    failRatio = 0;
}