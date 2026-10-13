#ifndef MONOSLAM_UTIL_H
#define MONOSLAM_UTIL_H

#include <opencv2/core.hpp>

namespace monoslam::util {

/*
 * Returns a square region of interest given its center pixel coordinates and size.
 * Note that only if size is an odd number will the square be truly centered at the
 * desired position.
 *
 * center    Center of the square
 * size      Size of the square
 */
cv::Rect
buildSquare(const cv::Point& center, int size);

/*
 * Removes the nth row and column of a square matrix. The matrix is not reduced
 * in-place, but a deep copy is returned.
 *
 * mat    Matrix to be reduced
 * idx    Row and column to be deleted
 */
cv::Mat
removeRowCol(const cv::Mat& mat, int idx);

/*
 * Removes specific rows and columns of a square matrix. The matrix is not reduced in-place,
 * but a deep copy is returned. The indices to be removed should be sorted in ascending order.
 * The offset specifies the shift applied to the indices. The stride indicates how the matrix
 * rows and columns are grouped together and referenced by the indices. For example, given
 * indices 0, 3, 7, an offset of 5 and a stride of 2, the deleted rows and columns are 5, 6,
 * 11, 12, 19, 20.
 *
 * mat        Matrix to be reduced
 * indices    Sorted rows and columns to be deleted
 * offset     Index offset
 * stride     Row and column stride
 */
cv::Mat
removeRowsCols(const cv::Mat& mat, const std::vector<int>& indices, int offset = 0, int stride = 1);

/*
 * Removes the nth row of a matrix. The matrix is not reduced in-place, but a deep
 * copy is returned.
 *
 * mat    Matrix to be reduced
 * idx    Row to be deleted
 */
cv::Mat
removeRow(const cv::Mat& mat, int idx);

/*
 * Removes specific rows of a matrix. The matrix is not reduced in-place, but a deep
 * copy is returned. The indices to be removed should be sorted in ascending order.
 * The offset specifies the shift applied to the indices. The stride indicates how
 * the matrix rows are grouped together and referenced by the indices. For example,
 * given indices 0, 3, 7, an offset of 5 and a stride of 2, the deleted rows are
 * 5, 6, 11, 12, 19, 20.
 *
 * mat        Matrix to be reduced
 * indices    Sorted rows to be deleted
 * offset     Index offset
 * stride     Row stride
 */
cv::Mat
removeRows(const cv::Mat& mat, const std::vector<int>& indices, int offset = 0, int stride = 1);

/*
 * Returns the uncertainty ellipses where the predicted in-sight features should be found with
 * high probability. In particular, the ellipses are constructed so that they are n-sigma
 * confidence regions for a 2D Gaussian distribution. For instance, n = 2.146 corresponds to a
 * 0.9 confidence level.
 *
 * means        Pixel locations of the predicted in-sight features
 * S            Innovation covariance matrix of the predicted in-sight features
 * numSigmas    Number of standard deviations for the confidence region boundary
 */
std::vector<cv::RotatedRect>
computeEllipses(const std::vector<cv::Point2d>& means, const cv::Mat_<double>& S, float numSigmas = 2.146f);

/*
 * Returns the smallest straight rectangle which contains the given ellipse. The computed
 * rectangle is cropped if it exceeds the given image boundary so that it fits inside it.
 *
 * ellipse      Ellipse to be bounded
 * imageSize    Size of the image
 */
cv::Rect
getBoundingBox(const cv::RotatedRect& ellipse, const cv::Size& imageSize);

/*
 * Draws a rectangle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * rectangle    Rectangle
 * color        Color of the rectangle
 */
void
drawRectangle(cv::Mat& image, const cv::Rect& rectangle, const cv::Scalar& color);

/*
 * Draws a square on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image     Grayscale or color image to draw on
 * center    Center of the square
 * width     Width of the square
 * color     Color of the square
 */
void
drawSquare(cv::Mat& image, const cv::Point& center, int width, const cv::Scalar& color);

/*
 * Draws an ellipse on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image      Grayscale or color image to draw on
 * ellipse    Ellipse
 * color      Color of the ellipse
 */
void
drawEllipse(cv::Mat& image, const cv::RotatedRect& ellipse, const cv::Scalar& color);

/*
 * Draws a circle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image     Grayscale or color image to draw on
 * center    Center of the circle
 * radius    Radius of the circle
 * color     Color of the circle
 */
void
drawCircle(cv::Mat& image, const cv::Point& center, int radius, const cv::Scalar& color);

/*
 * Draws a template on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image       Grayscale or color image to draw on
 * templ       Grayscale or color template to be drawn
 * position    Position of the origin of the template in the image
 * cx          Horizontal coordinate of the origin of the template, within the template
 * cy          Vertical coordinate of the origin of the template, within the template
 */
void
drawTemplate(cv::Mat& image, const cv::Mat& templ, const cv::Point2d& position, int cx, int cy);

/*
 * Computes the image corresponding to a given region of interest of a frame, frame(roi),
 * and extends it so that the template's center pixel coordinates u, v are superimposed over
 * all of the image pixels of frame(roi), when cv::matchTemplate is called with it and the template.
 * The provided region of interest is also modified accordingly. The goal is to make cv::matchTemplate
 * return a matrix that has exactly the same size as roi, where each pixel in the result represents
 * the match score if the template were perfectly centered on that pixel. This function addresses
 * two issues: (1) The default computed image, frame(roi), may be smaller than the template, hence
 * cv::matchTemplate cannot be used. (2) Even if frame(roi) is at least as large as the template,
 * it might be necessary to test the template against all the pixel positions of the image,
 * the former being centered at a specific pixel location (u, v). This is not how cv::matchTemplate
 * operates (it shifts the template around the image but always keeps it within its bounds), so
 * the image must be correctly resized to achieve this behavior.
 *
 * frame    Grayscale or color base image
 * templ    Grayscale or color template to match
 * u        First pixel coordinate of the patch matching center
 * v        Second pixel coordinate of the patch matching center
 * roi      Original region of interest
 */
cv::Mat
computeMatchingImage(const cv::Mat& frame, const cv::Mat& templ, int u, int v, cv::Rect& roi);

/*
 * 2D Gaussian probability density function.
 *
 * x       Point to evaluate the function on
 * mean    Mean
 * cov     Covariance
 */
double
gaussian2Dpdf(const cv::Vec2d& x, const cv::Vec2d& mean, const cv::Matx22d& cov);

/*
 * Removes specific elements of a vector. The vector is reduced in-place. The indices of the
 * elements to be removed should be sorted in ascending order.
 *
 * vec        Vector to be reduced
 * indices    Sorted indices of elements to be deleted
 */
template<class T>
void
removeIndices(std::vector<T>& vec, const std::vector<int>& indices)  // TODO: Is this fragile?
{
    std::size_t i = 0;  // Current position in vec
    std::size_t j = 0;  // Current position in indices

    auto it = std::remove_if(vec.begin(), vec.end(), [&](const T&) {
        const bool remove = (j < indices.size() && static_cast<std::size_t>(indices[j]) == i);

        if (remove) {
            ++j;
        }

        ++i;

        return remove;
    });

    vec.erase(it, vec.end());
}

}

#endif
