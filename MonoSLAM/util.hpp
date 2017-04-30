#ifndef UTIL_H
#define UTIL_H

#include "opencv2/core/core.hpp"

constexpr double PI         = 3.1415926535897932385;
constexpr double PI_DOUBLE  = 6.2831853071795864769;
constexpr double RAD_TO_DEG = 57.2957795130823208768;
constexpr double DEG_TO_RAD = 0.0174532925199432957;

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
cv::Mat computeHomography(const cv::Mat& p, const cv::Mat& n, const cv::Mat& K,
                          const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, const cv::Mat& t2);

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
cv::Mat buildDistortionMap(const cv::Size& size, const cv::Mat& K, const cv::Mat& distCoeffs);

/*
 * Returns a square region of interest given its center pixel coordinates and size.
 * Note that only if size is an odd number will the square be truly centered at the
 * desired position.
 */
cv::Rect buildSquare(const cv::Point2i& center, int size);

/*
 * Removes the nth row and column of a square matrix. The matrix is not reduced in-place,
 * but a deep copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row and column to be deleted
 */
cv::Mat reduceMat(const cv::Mat& mat, int idx);

std::vector<cv::RotatedRect> computeEllipses(const std::vector<cv::Point2d>& means, const cv::Mat& S);
cv::Rect getBoundingBox(const cv::RotatedRect& ellipse);
void drawEllipse(Mat& image, const cv::RotatedRect& ellipse);

#endif