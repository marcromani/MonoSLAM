#include <cmath>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

constexpr double PI_DOUBLE  = 6.2831853071795864769;
constexpr double RAD_TO_DEG = 57.295779513082320877;
constexpr double DEG_TO_RAD = 0.0174532925199432957;

vector<RotatedRect> computeEllipses(const vector<Point2d>& means, const Mat& S);
Rect getBoundingBox(const RotatedRect& ellipse, const Size& imageSize);
void drawEllipse(Mat& image, const RotatedRect& e, const Scalar& color);
void drawRectangle(Mat& image, const Rect& r, const Scalar& color);

int main() {

    // Canvas
    Size frameSize(1000, 700);
    Mat image(frameSize, CV_8UC3, Scalar::all(0));

    // Mean vector and covariance matrix
    Point2d mean(frameSize.width / 2, frameSize.height / 2);
    Mat cov = (Mat_<double>(2, 2) << 3500, 2900, 2900, 3000);

    // Compute the error ellipse
    vector<Point2d> means = {mean};
    vector<RotatedRect> ellipses = computeEllipses(means, cov);
    RotatedRect ellipse = ellipses[0];

    // Compute the ellipse bounding box
    Rect rect = getBoundingBox(ellipse, frameSize);

    // Show the result
    drawEllipse(image, ellipse, Scalar(0, 0, 255));
    drawRectangle(image, rect, Scalar(0, 255, 0));
    imshow("Ellipse", image);
    waitKey();
}

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
        double majorAxis = 6.06970851754 * sqrt(eigenval.at<double>(0));
        double minorAxis = 6.06970851754 * sqrt(eigenval.at<double>(1));

        ellipses.push_back(RotatedRect(means[i], Size(majorAxis, minorAxis), -angle));
    }

    return ellipses;
}

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

void drawEllipse(Mat& image, const RotatedRect& e, const Scalar& color) {

    ellipse(image, e, color, 1, LINE_AA);
}

void drawRectangle(Mat& image, const Rect& r, const Scalar& color) {

    rectangle(image, r, color, 1, LINE_AA);
}