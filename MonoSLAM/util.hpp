#ifndef UTIL_H
#define UTIL_H

#include "opencv2/core/core.hpp"

constexpr double PI         = 3.1415926535897932385;
constexpr double PI_DOUBLE  = 6.2831853071795864769;
constexpr double RAD_TO_DEG = 57.295779513082320877;
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
 * Returns a square region of interest given its center pixel coordinates and size.
 * Note that only if size is an odd number will the square be truly centered at the
 * desired position.
 *
 * center       Center of the square
 * size         Size of the square
 */
cv::Rect buildSquare(const cv::Point2i& center, int size);

/*
 * Removes the nth row and column of a square matrix. The matrix is not reduced
 * in-place, but a deep copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row and column to be deleted
 */
cv::Mat removeRowCol(const cv::Mat& mat, int idx);

/*
 * Removes specific rows and columns of a square matrix. The matrix is not reduced in-place,
 * but a deep copy is returned. The indices to be removed should be sorted in ascending order.
 * The offset specifies the shift applied to the indices. The stride indicates how the matrix
 * rows and columns are grouped together and referenced by the indices. For example, given
 * indices 0, 3, 7, an offset of 5 and a stride of 2, the deleted rows and columns are 5, 6,
 * 11, 12, 19, 20.
 *
 * mat          Matrix to be reduced
 * indices      Sorted rows and columns to be deleted
 * offset       Indices offset
 * stride       Row and column stride
 */
cv::Mat removeRowsCols(const cv::Mat& mat, std::vector<int>& indices, int offset = 0, int stride = 1);

/*
 * Removes the nth row of a matrix. The matrix is not reduced in-place, but a deep
 * copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row to be deleted
 */
cv::Mat removeRow(const cv::Mat& mat, int idx);

/*
 * Removes specific rows of a matrix. The matrix is not reduced in-place, but a deep
 * copy is returned. The indices to be removed should be sorted in ascending order.
 * The offset specifies the shift applied to the indices. The stride indicates how
 * the matrix rows are grouped together and referenced by the indices. For example,
 * given indices 0, 3, 7, an offset of 5 and a stride of 2, the deleted rows are
 * 5, 6, 11, 12, 19, 20.
 *
 * mat          Matrix to be reduced
 * indices      Sorted rows to be deleted
 * offset       Indices offset
 * stride       Row stride
 */
cv::Mat removeRows(const cv::Mat& mat, std::vector<int>& indices, int offset = 0, int stride = 1);

/*
 * Removes specific elements of a vector. The vector is reduced in-place. The indices of the
 * elements to be removed should be sorted in ascending order.
 *
 * vec          Vector to be reduced
 * indices      Sorted indices of elements to be deleted
 */
template <class T>
void removeIndices(std::vector<T>& vec, const std::vector<int>& indices) {

    for (int i = indices.size() - 1; i >= 0; i--)
        vec.erase(vec.begin() + indices[i]);
}

/*
 * Returns the smallest straight rectangle which contains the given ellipse. The computed
 * rectangle is cropped if it exceeds the given image boundary so that it fits inside it.
 *
 * ellipse      Ellipse to be bounded
 * imageSize    Size of the image
 */
cv::Rect getBoundingBox(const cv::RotatedRect& ellipse, const cv::Size& imageSize);

/*
 * Draws a rectangle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * rectangle    Rectangle
 * color        Color of the rectangle
 */
void drawRectangle(cv::Mat& image, const cv::Rect& rectangle, const cv::Scalar& color);

/*
 * Draws a square on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * center       Center of the square
 * width        Width of the square
 * color        Color of the square
 */
void drawSquare(cv::Mat& image, const cv::Point2i& center, int width, const cv::Scalar& color);

/*
 * Draws an ellipse on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * ellipse      Ellipse
 * color        Color of the ellipse
 */
void drawEllipse(cv::Mat& image, const cv::RotatedRect& ellipse, const cv::Scalar& color);

/*
 * Draws a circle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * center       Center of the circle
 * radius       Radius of the circle
 * color        Color of the circle
 */
void drawCircle(cv::Mat& image, const cv::Point2i& center, int radius, const cv::Scalar& color);

/*
 * Draws the template on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * templ        Grayscale or color template to be drawn
 * position     Image position of the template origin
 * cx           First pixel coordinate of the template origin
 * cy           Second pixel coordinate of the template origin
 */
void drawTemplate(cv::Mat& image, const cv::Mat& templ, const cv::Point2d& position, int cx, int cy);

/*
 * Computes the image corresponding to a given region of interest of a frame such that
 * the template pixel coordinates u, v are superimposed over all of the image pixels of
 * frame(roi) when matchTemplate is called for this image and the template. The provided
 * region of interest is modified conveniently. This function addresses several issues:
 * (1) The default computed image, frame(roi), may be smaller than the template, hence
 * matchTemplate can not be used. (2) Even if the image is at least as large as the template,
 * it might be necessary to test the template against all the pixel positions of the image,
 * the former being centered at a specific pixel location. This is not how matchTemplate
 * operates (it shifts the template around the image but keeps it always inside it) so
 * the image must be correctly resized to achieve this behavior.
 *
 * frame    Grayscale or color base image
 * templ    Grayscale or color template to match
 * u        First pixel coordinate of the patch matching center
 * v        Second pixel coordinate of the patch matching center
 * roi      Original region of interest
 */
cv::Mat computeMatchingImage(const cv::Mat& frame, const cv::Mat& templ, int u, int v, cv::Rect& roi);

#endif