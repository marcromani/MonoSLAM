#include "util.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "numbers.h"

cv::Rect
monoslam::util::buildSquare(const cv::Point& center, int size)
{
    const int left = center.x - size / 2;
    const int top = center.y - size / 2;

    return cv::Rect(left, top, size, size);
}

cv::Mat
monoslam::util::removeRowCol(const cv::Mat& mat, int idx)
{
    assert(mat.dims == 2);
    assert(mat.rows == mat.cols);

    const int n = mat.rows;

    assert(0 <= idx && idx < n);

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
monoslam::util::removeRowsCols(const cv::Mat& mat, const std::vector<int>& indices, int offset, int stride)
{
    assert(offset > 0 && stride > 0);

    cv::Mat result = mat.clone();

    int removed = 0;

    for (int index : indices) {
        const int idx = offset + stride * (index - removed);

        for (int j = 0; j < stride; ++j) {
            result = removeRowCol(result, idx);
        }

        ++removed;
    }

    return result;
}

cv::Mat
monoslam::util::removeRow(const cv::Mat& mat, int idx)
{
    assert(mat.dims == 2);
    assert(mat.rows == mat.cols);

    const int n = mat.rows;

    assert(0 <= idx && idx < n);

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
monoslam::util::removeRows(const cv::Mat& mat, const std::vector<int>& indices, int offset, int stride)
{
    assert(offset > 0 && stride > 0);

    cv::Mat result = mat.clone();

    int removed = 0;

    for (int index : indices) {
        const int idx = offset + stride * (index - removed);

        for (int j = 0; j < stride; ++j) {
            result = removeRow(result, idx);
        }

        ++removed;
    }

    return result;
}

std::vector<size_t>
monoslam::util::complement(const std::vector<size_t>& values, size_t max)
{
    assert(values.size() <= max);

    std::vector<size_t> other;
    other.reserve(max - values.size());

    size_t idx = 0;

    for (size_t i = 0; i < max; ++i) {
        if (idx < values.size() && i == values[idx]) {
            ++idx;

        } else {
            other.push_back(i);
        }
    }

    return other;
}

cv::Mat
monoslam::util::selectRows(const cv::Mat& mat, const std::vector<size_t>& rows)
{
    assert(mat.dims == 2);

    for (size_t r : rows) {
        assert(r < static_cast<size_t>(mat.rows));
    }

    cv::Mat result(static_cast<int>(rows.size()), mat.cols, mat.type());

    if (rows.empty()) {
        return result;
    }

    const bool contiguousRows =
      std::adjacent_find(rows.begin(), rows.end(), [](size_t a, size_t b) { return b != a + 1; }) == rows.end();

    if (contiguousRows) {
        mat.rowRange(static_cast<int>(rows.front()), static_cast<int>(rows.front() + rows.size())).copyTo(result);

    } else {
        for (size_t i = 0; i < rows.size(); ++i) {
            const int idx = static_cast<int>(rows[i]);

            mat.row(idx).copyTo(result.row(static_cast<int>(i)));
        }
    }

    return result;
}

cv::Mat
monoslam::util::selectCols(const cv::Mat& mat, const std::vector<size_t>& cols)
{
    assert(mat.dims == 2);

    for (size_t c : cols) {
        assert(c < static_cast<size_t>(mat.cols));
    }

    cv::Mat result(mat.rows, static_cast<int>(cols.size()), mat.type());

    if (cols.empty()) {
        return result;
    }

    const bool contiguousCols =
      std::adjacent_find(cols.begin(), cols.end(), [](size_t a, size_t b) { return b != a + 1; }) == cols.end();

    if (contiguousCols) {
        mat.colRange(static_cast<int>(cols.front()), static_cast<int>(cols.front() + cols.size())).copyTo(result);

    } else {
        const size_t elemSize = mat.elemSize();

        for (int i = 0; i < mat.rows; ++i) {
            const std::byte* src = mat.ptr<std::byte>(i);

            std::byte* dst = result.ptr<std::byte>(i);

            for (size_t j = 0; j < cols.size(); ++j) {
                std::memcpy(dst + j * elemSize, src + cols[j] * elemSize, elemSize);
            }
        }
    }

    return result;
}

cv::Mat
monoslam::util::selectRowsCols(const cv::Mat& mat, const std::vector<size_t>& rows, const std::vector<size_t>& cols)
{
    assert(mat.dims == 2);

    for (size_t r : rows) {
        assert(r < static_cast<size_t>(mat.rows));
    }

    for (size_t c : cols) {
        assert(c < static_cast<size_t>(mat.cols));
    }

    cv::Mat result(static_cast<int>(rows.size()), static_cast<int>(cols.size()), mat.type());

    if (rows.empty() || cols.empty()) {
        return result;
    }

    const size_t elemSize = mat.elemSize();

    const bool contiguousCols =
      std::adjacent_find(cols.begin(), cols.end(), [](size_t a, size_t b) { return b != a + 1; }) == cols.end();

    if (contiguousCols) {
        const bool contiguousRows =
          std::adjacent_find(rows.begin(), rows.end(), [](size_t a, size_t b) { return b != a + 1; }) == rows.end();

        if (contiguousRows) {
            mat(cv::Rect(static_cast<int>(cols.front()),
                         static_cast<int>(rows.front()),
                         static_cast<int>(cols.size()),
                         static_cast<int>(rows.size())))
              .copyTo(result);

        } else {
            for (size_t i = 0; i < rows.size(); ++i) {
                const std::byte* src = mat.ptr<std::byte>(static_cast<int>(rows[i]));

                std::byte* dst = result.ptr<std::byte>(static_cast<int>(i));

                std::memcpy(dst, src + cols.front() * elemSize, cols.size() * elemSize);
            }
        }

    } else {
        for (size_t i = 0; i < rows.size(); ++i) {
            const std::byte* src = mat.ptr<std::byte>(static_cast<int>(rows[i]));

            std::byte* dst = result.ptr<std::byte>(static_cast<int>(i));

            for (size_t j = 0; j < cols.size(); ++j) {
                std::memcpy(dst + j * elemSize, src + cols[j] * elemSize, elemSize);
            }
        }
    }

    return result;
}

std::vector<cv::RotatedRect>
monoslam::util::computeEllipses(const std::vector<cv::Point2d>& means, const cv::Mat_<double>& S, float numSigmas)
{
    std::vector<cv::RotatedRect> ellipses;
    ellipses.reserve(means.size());

    const float diameterFactor = 2.0f * numSigmas;

    for (size_t i = 0; i < means.size(); ++i) {
        const int idx = static_cast<int>(2 * i);

        const cv::Matx22f Si =
          static_cast<cv::Matx22f>(S(cv::Rect(idx, idx, 2, 2)));  // Calls OpenCV's saturate_cast under the hood

        // Compute the eigenvalues and eigenvectors
        cv::Vec2f eigenvals;
        cv::Matx22f eigenvecs;
        cv::eigen(Si, eigenvals, eigenvecs);

        // Compute the angle between the largest eigenvector and the x axis
        float angle = std::atan2(eigenvecs(0, 1), eigenvecs(0, 0));

        // Shift the angle from [-pi, pi] to [0, 2pi]
        if (angle < 0.0f) {
            angle += numbers::PI_DOUBLE_F;
        }

        // Convert to degrees
        angle *= numbers::RAD_TO_DEG_F;

        // Compute the size of the major and minor axes
        const float majorAxis = diameterFactor * std::sqrt(eigenvals[0]);
        const float minorAxis = diameterFactor * std::sqrt(eigenvals[1]);

        ellipses.push_back(
          cv::RotatedRect(static_cast<cv::Point2f>(means[i]), cv::Size2f(majorAxis, minorAxis), -angle));
    }

    return ellipses;
}

cv::Rect
monoslam::util::getBoundingBox(const cv::RotatedRect& ellipse, const cv::Size& imageSize)
{
    const double angle = ellipse.angle * numbers::DEG_TO_RAD;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = 0.5 * ellipse.size.width;
    const double b = 0.5 * ellipse.size.height;

    const double xmax = std::sqrt(a * a * c * c + b * b * s * s);
    const double ymax = std::sqrt(a * a * s * s + b * b * c * c);

    const cv::Rect box(static_cast<int>(std::floor(ellipse.center.x - xmax)),
                       static_cast<int>(std::floor(ellipse.center.y - ymax)),
                       static_cast<int>(std::ceil(2.0 * xmax)),
                       static_cast<int>(std::ceil(2.0 * ymax)));

    // TODO: Could we use this anywhere else?
    return box & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

void
monoslam::util::drawRectangle(cv::Mat& image, const cv::Rect& r, const cv::Scalar& color)
{
    cv::rectangle(image, r, color, 1, cv::LINE_AA);
}

void
monoslam::util::drawSquare(cv::Mat& image, const cv::Point& center, int width, const cv::Scalar& color)
{
    cv::rectangle(image, buildSquare(center, width), color, 1, cv::LINE_AA);
}

void
monoslam::util::drawEllipse(cv::Mat& image, const cv::RotatedRect& e, const cv::Scalar& color)
{
    cv::ellipse(image, e, color, 1, cv::LINE_AA);
}

void
monoslam::util::drawCircle(cv::Mat& image, const cv::Point& center, int radius, const cv::Scalar& color)
{
    cv::circle(image, center, radius, color, 1, cv::LINE_AA);
}

void
monoslam::util::drawTemplate(cv::Mat& image, const cv::Mat& templ, const cv::Point2d& position, int cx, int cy)
{
    // TODO: Can we use the cv::Rect intersection here?
    int templ_x = 0;
    int image_x = static_cast<int>(position.x - cx);

    if (image_x > image.cols - 1) {
        return;
    }

    if (image_x < 0) {
        templ_x = -image_x;
        image_x = 0;
    }

    int templ_y = 0;
    int image_y = static_cast<int>(position.y - cy);

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
monoslam::util::computeMatchingImage(const cv::Mat& frame, const cv::Mat& templ, int u, int v, cv::Rect& roi)
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
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0.0));

    return image;
}

double
monoslam::util::gaussian2D(const cv::Vec2d& x, const cv::Vec2d& mean, const cv::Matx22d& cov)
{
    const cv::Vec2d diff = x - mean;

    const double dx = diff[0];
    const double dy = diff[1];

    const double a = cov(0, 0);
    const double b = cov(0, 1);
    const double c = cov(1, 0);
    const double d = cov(1, 1);

    const double det = a * d - b * c;
    assert(det > 1e-12);  // TODO: Should have a global epsilon

    const double inv00 = d / det;
    const double inv01 = -b / det;
    const double inv10 = -c / det;
    const double inv11 = a / det;

    const double mahalanobis2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

    return std::exp(-0.5 * mahalanobis2) / (numbers::PI_DOUBLE * std::sqrt(det));
}
