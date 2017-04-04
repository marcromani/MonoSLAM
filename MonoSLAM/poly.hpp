#ifndef POLY_H
#define POLY_H

#include <vector>

#include "opencv2/core/core.hpp"

class Polyhedron {

public:

    std::vector<cv::Point3d>    vertices;
    std::vector<cv::Point3i>    faces;

    std::vector<cv::Point2i>    edges;

    Polyhedron(const std::vector<cv::Point3d>&, const std::vector<cv::Point3i>&);

    Polyhedron intersect(const Polyhedron&);
};

#endif