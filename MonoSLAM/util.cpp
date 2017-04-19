#include "util.hpp"

using namespace cv;

/*
 * Computes the homography matrix H between two pinhole camera views of a plane.
 * In particular, p2 = H * p1, where p1 is the pixel in the first view which
 * corresponds to a point p in the plane, and p2 is the pixel which corresponds
 * to p in the second view.
 *
 * p    Point in the plane, in world coordinates
 * n    Plane normal, in world coordinates
 * K    Camera intrinsic matrix
 * R1   First pose rotation (world to camera)
 * t1   First pose translation, in world coordinates
 * R2   Second pose rotation (world to camera)
 * t2   Second pose translation, in world coordinates
 */
Mat computeHomography(const Mat& p, const Mat& n, const Mat& K,
                      const Mat& R1, const Mat& t1, const Mat& R2, const Mat& t2) {

    Mat m = n.t() * (p - t1);
    Mat H = K * R2 * (m.at<double>(0, 0) * Mat::eye(3, 3, CV_64FC1) + (t1 - t2) * n.t()) * R1.t() * K.inv();

    return H;
}

/*
 * Builds a distortion map M which relates the image pixels of a pinhole camera
 * model to the image pixels of the same camera when a lens distortion model is
 * added to it. In particular, (u, v) = M(du, dv), where (u, v) is a pixel in the
 * pinhole camera model and (du, dv) is a pixel in the extended camera model. At
 * the moment, only 3 radial distortion coefficients are supported.
 *
 * size         Image size
 * K            Camera intrinsic matrix
 * distCoeffs   Distortion coefficients
 */
Mat buildDistortionMap(const Size& size, const Mat& K, const Mat& distCoeffs) {

    // Init map, distorted pixels with no relative undistorted pixel are mapped to (-1, -1)
    Mat map = Mat(size.height, size.width, CV_32FC2, Scalar(-1, -1));

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(0, 1);
    double p1 = distCoeffs.at<double>(0, 2);
    double p2 = distCoeffs.at<double>(0, 3);
    double k3 = distCoeffs.at<double>(0, 4);

    for (int u = 0; u < 5*size.width; u++) {
        for (int v = 0; v < 5*size.height; v++) {

            double u_ = 0.2*u;
            double v_ = 0.2*v;

            double x = (u_ - cx) / fx;
            double y = (v_ - cy) / fy;

            double r2 = x*x + y*y;

            double xx = x*(1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2*p1*x*y + p2*(r2 + 2*x*x);
            double yy = y*(1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + p1*(r2 + 2*y*y) + 2*p2*x*y;

            int du = (int) round(fx * xx + cx);
            int dv = (int) round(fy * yy + cy);

            if (0 <= du && du < size.width && 0 <= dv && dv < size.height) {

                map.at<Vec2f>(dv, du)[0] = u_;
                map.at<Vec2f>(dv, du)[1] = v_;
            }
        }
    }

    return map;
}

/*
 * Returns a square region of interest given its center pixel coordinates and size.
 * Note that only if size is an odd number will the square be truly centered at the
 * desired position.
 */
Rect buildSquare(const Point2i& center, int size) {

    int left = center.x - size / 2;
    int top = center.y - size / 2;

    return Rect(left, top, size, size);
}

/*
 * Removes the nth row and column of a square matrix. The matrix is not reduced in-place,
 * but a deep copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row and column to be deleted
 */
Mat reduceMat(const Mat& mat, int idx) {

    int n = mat.rows;

    Mat reduced(n - 1, n - 1, mat.type());

    if (idx == 0)
        mat(Rect(1, 1, n - 1, n - 1)).copyTo(reduced(Rect(0, 0, n - 1, n - 1)));

    else if (idx == n - 1)
        mat(Rect(0, 0, n - 1, n - 1)).copyTo(reduced(Rect(0, 0, n - 1, n - 1)));

    else {
        mat(Rect(0, 0, idx, idx)).copyTo(reduced(Rect(0, 0, idx, idx)));
        mat(Rect(idx+1, 0, n-idx-1, idx)).copyTo(reduced(Rect(idx, 0, n-idx-1, idx)));
        mat(Rect(0, idx+1, idx, n-idx-1)).copyTo(reduced(Rect(0, idx, idx, n-idx-1)));
        mat(Rect(idx+1, idx+1, n-idx-1, n-idx-1)).copyTo(reduced(Rect(idx, idx, n-idx-1, n-idx-1)));
    }

    return reduced;
}