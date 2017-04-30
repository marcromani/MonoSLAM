#include <cmath>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

RotatedRect getEllipse(double prob, const Point2d& mean, const Mat& cov);
Rect getBoundingBox(const RotatedRect& ellipse);
void drawEllipse(Mat& image, const RotatedRect& ellipse);

int main() {

    // Mean vector
    Point2d mean(500, 350);

    // Covariance matrix
    Mat cov = (Mat_<double>(2, 2) << 1100, 1000, 1000, 2500);

    // Compute the error ellipse for a given confidence interval
    RotatedRect ellipse = getEllipse(0.99, mean, cov);

    // Show the result
    Mat image(700, 1000, CV_8UC3, Scalar::all(0));

    drawEllipse(image, ellipse);

    imshow("Ellipse", image);
    waitKey();
}

RotatedRect getEllipse(double prob, const Point2d& mean, const Mat& cov) {

    // Compute the eigenvalues and eigenvectors
    Mat eigenval, eigenvec;
    eigen(cov, eigenval, eigenvec);

    // Compute the angle between the largest eigenvector and the x axis
    double angle = atan2(eigenvec.at<double>(0, 1), eigenvec.at<double>(0, 0));

    // Shift the angle from [-pi, pi] to [0, 2pi]
    if (angle < 0)
        angle += 6.28318530718;

    // Convert to degrees
    angle *= 180 / 3.14159265359;

    // Compute the chi-squared value such that P(X <= chiSquaredVal) = prob
    double chiSquaredVal = - 2 * log(1 - prob);

    // Compute the size of the major and minor axes
    double majorAxis = 2 * sqrt(chiSquaredVal * eigenval.at<double>(0));
    double minorAxis = 2 * sqrt(chiSquaredVal * eigenval.at<double>(1));

    // OpenCV defines the angle clockwise instead of anticlockwise
    return RotatedRect(mean, Size(majorAxis, minorAxis), -angle);
}

Rect getBoundingBox(const RotatedRect& ellipse) {

    double angle = ellipse.angle * 3.14159265359 / 180;
    double x = ellipse.center.x;
    double y = ellipse.center.y;
    double a = 0.5 * ellipse.size.width;
    double b = 0.5 * ellipse.size.height;

    double xmax, ymax;

    if (a == b) {

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

    return Rect(x - xmax, y - ymax, 2 * xmax, 2 * ymax);
}

void drawEllipse(Mat& image, const RotatedRect& ellipse_) {

    // Draw the ellipse
    ellipse(image, ellipse_, Scalar::all(255), 1);

    // Draw the bounding box
    rectangle(image, getBoundingBox(ellipse_), Scalar(0, 255, 0));
}