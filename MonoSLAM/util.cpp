#include <algorithm>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"

#include "util.hpp"

using namespace cv;
using namespace std;

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
 * Returns a square region of interest given its center pixel coordinates and size.
 * Note that only if size is an odd number will the square be truly centered at the
 * desired position.
 *
 * center       Center of the square
 * size         Size of the square
 */
Rect buildSquare(const Point2i& center, int size) {

    int left = center.x - size / 2;
    int top = center.y - size / 2;

    return Rect(left, top, size, size);
}

/*
 * Removes the nth row and column of a square matrix. The matrix is not reduced
 * in-place, but a deep copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row and column to be deleted
 */
Mat removeRowCol(const Mat& mat, int idx) {

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
Mat removeRowsCols(const Mat& mat, vector<int>& indices, int offset, int stride) {

    Mat result = mat.clone();

    for (unsigned int i = 0; i < indices.size(); i++) {

        int idx = offset + stride * (indices[i] - i);

        for (int j = 0; j < stride; j++)
            result = removeRowCol(result, idx);
    }

    return result;
}

/*
 * Removes the nth row of a matrix. The matrix is not reduced in-place, but a deep
 * copy is returned.
 *
 * mat      Matrix to be reduced
 * idx      Row to be deleted
 */
Mat removeRow(const Mat& mat, int idx) {

    Mat result;

    int n = mat.rows;

    if (idx == 0)
        result = mat.rowRange(1, n).clone();

    else if (idx == n - 1)
        result = mat.rowRange(0, n - 1).clone();

    else
        vconcat(mat.rowRange(0, idx), mat.rowRange(idx + 1, n), result);

    return result;
}

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
Mat removeRows(const Mat& mat, vector<int>& indices, int offset, int stride) {

    Mat result = mat.clone();

    for (unsigned int i = 0; i < indices.size(); i++) {

        int idx = offset + stride * (indices[i] - i);

        for (int j = 0; j < stride; j++)
            result = removeRow(result, idx);
    }

    return result;
}

/*
 * Returns the ellipses where the predicted in sight features should be found with
 * high probability. In particular, the ellipses are constructed so that they are
 * confidence regions at level 0.86.
 *
 * means    Predicted in sight features pixel locations
 * S        Innovation covariance matrix of the predicted in sight features
 */
vector<RotatedRect> computeEllipses(const vector<Point2d>& means, const Mat& S) {

    vector<RotatedRect> ellipses;

    for (unsigned int i = 0; i < means.size(); i++) {

        Mat Si = S(Rect(2*i, 2*i, 2, 2));

        // Compute the eigenvalues and eigenvectors
        Mat eigenval, eigenvec;
        eigen(Si, eigenval, eigenvec);

        // Compute the angle between the largest eigenvector and the x axis
        double angle = atan2(eigenvec.at<double>(0, 1), eigenvec.at<double>(0, 0));

        // Shift the angle from [-pi, pi] to [0, 2pi]
        if (angle < 0)
            angle += PI_DOUBLE;

        // Convert to degrees
        angle *= RAD_TO_DEG;

        // Compute the size of the major and minor axes
        double majorAxis = 4 * sqrt(eigenval.at<double>(0));
        double minorAxis = 4 * sqrt(eigenval.at<double>(1));

        ellipses.push_back(RotatedRect(means[i], Size(majorAxis, minorAxis), -angle));
    }

    return ellipses;
}

/*
 * Returns the smallest straight rectangle which contains the given ellipse. The computed
 * rectangle is cropped if it exceeds the given image boundary so that it fits inside it.
 *
 * ellipse      Ellipse to be bounded
 * imageSize    Size of the image
 */
Rect getBoundingBox(const RotatedRect& ellipse, const Size& imageSize) {

    double angle = ellipse.angle * DEG_TO_RAD;
    double x = ellipse.center.x;
    double y = ellipse.center.y;
    double a = 0.5 * ellipse.size.width;
    double b = 0.5 * ellipse.size.height;

    double xmax, ymax;

    if (angle == 0 || a == b) {

        xmax = a;
        ymax = b;

    } else {

        double a2inv = 1 / (a*a);
        double b2inv = 1 / (b*b);

        double c = cos(angle);
        double s = sin(angle);
        double c2 = c*c;
        double s2 = s*s;

        double A = c2 * a2inv + s2 * b2inv;
        double B = c2 * b2inv + s2 * a2inv;
        double C = 2 * c * s * (b2inv - a2inv);

        double r = (4 * A * B / (C * C) - 1);
        double x0 = 1 / sqrt(A * r);
        double y0 = 1 / sqrt(B * r);

        if (C < 0) {

            x0 = - x0;
            y0 = - y0;
        }

        xmax = 2 * B * y0 / C;
        ymax = 2 * A * x0 / C;
    }

    double x0 = x - xmax;
    double y0 = y - ymax;
    double w = 2 * xmax + 1;
    double h = 2 * ymax + 1;

    if (x0 + w > imageSize.width)
        w = imageSize.width - x0;
    if (y0 + h > imageSize.height)
        h = imageSize.height - y0;

    return Rect(max(x - xmax, 0.), max(y - ymax, 0.), w, h);
}

/*
 * Draws a rectangle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * r            Rectangle
 * color        Color of the rectangle
 */
void drawRectangle(Mat& image, const Rect& r, const Scalar& color) {

    rectangle(image, r, color, 1, LINE_AA);
}

/*
 * Draws a square on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * center       Center of the square
 * width        Width of the square
 * color        Color of the square
 */
void drawSquare(Mat& image, const Point2i& center, int width, const Scalar& color) {

    rectangle(image, buildSquare(center, width), color, 1, LINE_AA);
}

/*
 * Draws an ellipse on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * e            Ellipse
 * color        Color of the ellipse
 */
void drawEllipse(Mat& image, const RotatedRect& e, const Scalar& color) {

    ellipse(image, e, color, 1, LINE_AA);
}

/*
 * Draws a circle on the given image. The image is modified by the function so a deep
 * copy should be made in order to preserve the original.
 *
 * image        Grayscale or color image to draw on
 * center       Center of the circle
 * radius       Radius of the circle
 * color        Color of the circle
 */
void drawCircle(Mat& image, const Point2i& center, int radius, const Scalar& color) {

    circle(image, center, radius, color, 1, LINE_AA);
}

void drawTemplate(Mat& image, const Mat& templ, const Point2d& position, int cx, int cy) {

    int templ_x = 0;
    int image_x = position.x - cx;

    if (image_x > image.cols - 1)
        return;

    if (image_x < 0) {

        templ_x = - image_x;
        image_x = 0;
    }

    int templ_y = 0;
    int image_y = position.y - cy;

    if (image_y > image.rows - 1)
        return;

    if (image_y < 0) {

        templ_y = - image_y;
        image_y = 0;
    }

    int w = templ.cols - templ_x;

    if (w <= 0)
        return;

    if (image_x + w > image.cols)
        w = image.cols - image_x;

    int h = templ.rows - templ_y;

    if (h <= 0)
        return;

    if (image_y + h > image.rows)
        h = image.rows - image_y;

    templ(Rect(templ_x, templ_y, w, h)).copyTo(image(Rect(image_x, image_y, w, h)));
}