#include "poly.hpp"

using namespace cv;
using namespace std;

namespace poly {

    int hash(const Point2i& p) {

        int x = p.x;
        int y = p.y;

        if (x < y) {

            x = y;
            y = p.x;
        }

        int v = x * (x - 1) / 2;

        v += x % 2 ? 2 * x - y : x + y;

        return v;
    }

    bool valid(const Point2i& p) {

        return p.x > -1 || p.y > -1;
    }
}

Polyhedron::Polyhedron(const vector<Point3d>& vertices_, const vector<Point3i>& faces_) {

    vertices = vertices_;
    faces = faces_;

    vector<Point2i> tmp(3 * faces.size(), Point2i(-1, -1));

    int idx;

    for (unsigned int i = 0; i < faces.size(); i++) {

        int x = faces[i].x;
        int y = faces[i].y;
        int z = faces[i].z;

        idx = poly::hash(Point2i(x, y));
        tmp[idx] = Point2i(x, y);

        idx = poly::hash(Point2i(x, z));
        tmp[idx] = Point2i(x, z);

        idx = poly::hash(Point2i(y, z));
        tmp[idx] = Point2i(y, z);
    }

    copy_if(begin(tmp), end(tmp), back_inserter(edges), poly::valid);
}

Polyhedron Polyhedron::intersect(const Polyhedron&) {

    vector<Point3d> vertices_;
    vector<Point3i> faces_;

    return Polyhedron(vertices_, faces_);
}