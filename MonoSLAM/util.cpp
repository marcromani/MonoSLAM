#include "util.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

cv::Matx33d
computeHomography(const cv::Matx31d& p,
                  const cv::Matx31d& n,
                  const cv::Matx33d& K,
                  const cv::Matx33d& R1,
                  const cv::Matx31d& t1,
                  const cv::Matx33d& R2,
                  const cv::Matx31d& t2)
{
    const cv::Matx<double, 1, 1> m = n.t() * (p - t1);
    const cv::Matx33d H = K * R2 * (m(0, 0) * cv::Matx33d::eye() + (t1 - t2) * n.t()) * R1.t() * K.inv();

    return H;
}

cv::Rect
buildSquare(const cv::Point& center, int size)
{
    const int left = center.x - size / 2;
    const int top = center.y - size / 2;

    return cv::Rect(left, top, size, size);
}

cv::Mat
removeRowCol(const cv::Mat& mat, int idx)
{
    CV_DbgAssert(mat.rows == mat.cols);

    const int n = mat.rows;

    CV_DbgAssert(0 <= idx && idx < n);

    if (n == 1) {
        return cv::Mat(0, 0, mat.type());
    }

    cv::Mat reduced(n - 1, n - 1, mat.type());

    if (idx == 0) {
        mat(cv::Rect(1, 1, n - 1, n - 1)).copyTo(reduced);

    } else if (idx == n - 1) {
        mat(cv::Rect(0, 0, n - 1, n - 1)).copyTo(reduced);

    } else {
        mat(cv::Rect(0, 0, idx, idx)).copyTo(reduced(cv::Rect(0, 0, idx, idx)));
        mat(cv::Rect(idx + 1, 0, n - idx - 1, idx)).copyTo(reduced(cv::Rect(idx, 0, n - idx - 1, idx)));
        mat(cv::Rect(0, idx + 1, idx, n - idx - 1)).copyTo(reduced(cv::Rect(0, idx, idx, n - idx - 1)));
        mat(cv::Rect(idx + 1, idx + 1, n - idx - 1, n - idx - 1))
          .copyTo(reduced(cv::Rect(idx, idx, n - idx - 1, n - idx - 1)));
    }

    return reduced;
}

cv::Mat
removeRowsCols(const cv::Mat& mat, const std::vector<int>& indices, int offset, int stride)
{
    CV_DbgAssert(offset > 0 && stride > 0);

    cv::Mat result = mat.clone();

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        const int idx = offset + stride * (indices[i] - i);

        for (int j = 0; j < stride; ++j) {
            result = removeRowCol(result, idx);
        }
    }

    return result;
}

cv::Mat
removeRow(const cv::Mat& mat, int idx)
{
    CV_DbgAssert(mat.rows == mat.cols);

    const int n = mat.rows;

    CV_DbgAssert(0 <= idx && idx < n);

    if (n == 1) {
        return cv::Mat(0, 0, mat.type());
    }

    cv::Mat result;

    if (idx == 0) {
        result = mat.rowRange(1, n).clone();

    } else if (idx == n - 1) {
        result = mat.rowRange(0, n - 1).clone();

    } else {
        cv::vconcat(mat.rowRange(0, idx), mat.rowRange(idx + 1, n), result);
    }

    return result;
}

cv::Mat
removeRows(const cv::Mat& mat, const std::vector<int>& indices, int offset, int stride)
{
    CV_DbgAssert(offset > 0 && stride > 0);

    cv::Mat result = mat.clone();

    for (unsigned int i = 0; i < static_cast<int>(indices.size()); ++i) {
        const int idx = offset + stride * (indices[i] - i);

        for (int j = 0; j < stride; ++j) {
            result = removeRow(result, idx);
        }
    }

    return result;
}

std::vector<cv::RotatedRect>
computeEllipses(const std::vector<cv::Point2d>& means, const cv::Mat_<double>& S)
{
    std::vector<cv::RotatedRect> ellipses;

    for (int i = 0; i < static_cast<int>(means.size()); ++i) {
        const cv::Matx22f Si =
          static_cast<cv::Matx22f>(S(cv::Rect(2 * i, 2 * i, 2, 2)));  // Calls OpenCV's saturate_cast under the hood

        // Compute the eigenvalues and eigenvectors
        cv::Vec2f eigenvals;
        cv::Matx22f eigenvecs;
        cv::eigen(Si, eigenvals, eigenvecs);

        // Compute the angle between the largest eigenvector and the x axis
        float angle = std::atan2(eigenvecs(0, 1), eigenvecs(0, 0));

        // Shift the angle from [-pi, pi] to [0, 2pi]
        if (angle < 0) {
            angle += PI_DOUBLE_F;
        }

        // Convert to degrees
        angle *= RAD_TO_DEG_F;

        // Compute the size of the major and minor axes
        const float majorAxis = 4 * std::sqrt(eigenvals(0));
        const float minorAxis = 4 * std::sqrt(eigenvals(1));

        ellipses.push_back(
          cv::RotatedRect(static_cast<cv::Point2f>(means[i]), cv::Size2f(majorAxis, minorAxis), -angle));
    }

    return ellipses;
}

cv::Rect
getBoundingBox(const cv::RotatedRect& ellipse, const cv::Size& imageSize)
{
    const double angle = ellipse.angle * DEG_TO_RAD;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = 0.5 * ellipse.size.width;
    const double b = 0.5 * ellipse.size.height;

    const double xmax = std::sqrt(a * a * c * c + b * b * s * s);
    const double ymax = std::sqrt(a * a * s * s + b * b * c * c);

    const Rect box(static_cast<int>(std::floor(ellipse.center.x - xmax)),
                   static_cast<int>(std::floor(ellipse.center.y - ymax)),
                   static_cast<int>(std::ceil(2 * xmax)),
                   static_cast<int>(std::ceil(2 * ymax)));

    // TODO: Could we use this anywhere else?
    return box & Rect(0, 0, imageSize.width, imageSize.height);
}

void
drawRectangle(cv::Mat& image, const cv::Rect& r, const cv::Scalar& color)
{
    cv::rectangle(image, r, color, 1, cv::LINE_AA);
}

void
drawSquare(cv::Mat& image, const cv::Point& center, int width, const cv::Scalar& color)
{
    cv::rectangle(image, buildSquare(center, width), color, 1, cv::LINE_AA);
}

void
drawEllipse(cv::Mat& image, const cv::RotatedRect& e, const cv::Scalar& color)
{
    cv::ellipse(image, e, color, 1, cv::LINE_AA);
}

void
drawCircle(cv::Mat& image, const cv::Point& center, int radius, const cv::Scalar& color)
{
    cv::circle(image, center, radius, color, 1, cv::LINE_AA);
}

void
drawTemplate(cv::Mat& image, const cv::Mat& templ, const cv::Point2d& position, int cx, int cy)
{
    // TODO: Can we use the cv::Rect intersection here?
    int templ_x = 0;
    int image_x = position.x - cx;

    if (image_x > image.cols - 1) {
        return;
    }

    if (image_x < 0) {
        templ_x = -image_x;
        image_x = 0;
    }

    int templ_y = 0;
    int image_y = position.y - cy;

    if (image_y > image.rows - 1) {
        return;
    }

    if (image_y < 0) {
        templ_y = -image_y;
        image_y = 0;
    }

    int w = templ.cols - templ_x;

    if (w <= 0) {
        return;
    }

    if (image_x + w > image.cols) {
        w = image.cols - image_x;
    }

    int h = templ.rows - templ_y;

    if (h <= 0) {
        return;
    }

    if (image_y + h > image.rows) {
        h = image.rows - image_y;
    }

    templ(cv::Rect(templ_x, templ_y, w, h)).copyTo(image(cv::Rect(image_x, image_y, w, h)));
}

cv::Mat
computeMatchingImage(const cv::Mat& frame, const cv::Mat& templ, int u, int v, cv::Rect& roi)
{
    // Compute the (inclusive) limits of the extended roi
    int x0 = roi.x - u;
    int y0 = roi.y - v;
    int x1 = roi.x + roi.width + templ.cols - u - 2;
    int y1 = roi.y + roi.height + templ.rows - v - 2;

    // Ensure that the roi coordinates are inside the frame boundaries
    // and compute how the image should be extended along its borders

    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;

    if (x0 < 0) {
        left = -x0;
        x0 = 0;
    }

    if (x1 > frame.cols - 1) {
        right = x1 - frame.cols + 1;
        x1 = frame.cols - 1;
    }

    if (y0 < 0) {
        top = -y0;
        y0 = 0;
    }

    if (y1 > frame.rows - 1) {
        bottom = y1 - frame.rows + 1;
        y1 = frame.rows - 1;
    }

    // Update the region of interest
    roi.x = x0;
    roi.y = y0;
    roi.width = x1 - x0 + 1;
    roi.height = y1 - y0 + 1;

    // Get the image
    cv::Mat image = frame(roi);

    // Extend the image with black borders
    cv::copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, cv::Scalar::all(0.0));

    return image;
}

double
gaussian2Dpdf(const cv::Matx21d& x, const cv::Matx21d& mean, const cv::Matx22d& cov)
{
    const double dx = x(0) - mean(0);
    const double dy = x(1) - mean(1);

    const double a = cov(0, 0);
    const double b = cov(0, 1);
    const double c = cov(1, 0);
    const double d = cov(1, 1);

    const double det = a * d - b * c;
    CV_DbgAssert(det > 1e-12);  // TODO: Should have a global epsilon

    const double inv00 = d / det;
    const double inv01 = -b / det;
    const double inv10 = -c / det;
    const double inv11 = a / det;

    const double mahalanobis2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

    return std::exp(-0.5 * mahalanobis2) / (PI_DOUBLE * std::sqrt(det));
}
