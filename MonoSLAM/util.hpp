#ifndef UTIL_H
#define UTIL_H

#include "opencv2/core/core.hpp"

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

#endif