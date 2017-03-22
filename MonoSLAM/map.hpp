#ifndef MAP_H
#define MAP_H

#include <vector>

#include "opencv2/core/core.hpp"

#include "camera.hpp"
#include "feature.hpp"

class Map {

public:

    Camera camera;                      // Camera
    std::vector<Feature> features;      // Initialized map features
    std::vector<Feature> candidates;    // Pre-initialized features

    cv::Mat x;                          // Map state (camera and features states)
    cv::Mat P;                          // Map state covariance matrix

    Map(const Camera& camera, const std::vector<Features>& features, const std::vector<Features>& candidates,
        const Mat& x, const Mat& P);

    Map();

    initMap()
};

#endif