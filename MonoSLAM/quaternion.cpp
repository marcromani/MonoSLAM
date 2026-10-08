#include "quaternion.hpp"

#include <cmath>

cv::Matx41d
computeQuaternion(const cv::Matx31d& rotv)
{
    const double x = rotv(0);
    const double y = rotv(1);
    const double z = rotv(2);

    const double angle = std::sqrt(x * x + y * y + z * z);

    const double c = cos(0.5 * angle);
    const double s = sin(0.5 * angle);

    return cv::Matx41d(c, s * x / angle, s * y / angle, s * z / angle);
}

cv::Matx41d
quaternionMultiply(const cv::Matx41d& q1, const cv::Matx41d& q2)
{
    const double a1 = q1(0);
    const double b1 = q1(1);
    const double c1 = q1(2);
    const double d1 = q1(3);

    const double a2 = q2(0);
    const double b2 = q2(1);
    const double c2 = q2(2);
    const double d2 = q2(3);

    return cv::Matx41d(a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                       a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                       a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                       a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2);
}

cv::Matx33d
getRotationMatrix(const cv::Matx41d& q)
{
    const double a = q(0);
    const double b = q(1);
    const double c = q(2);
    const double d = q(3);

    return cv::Matx33d(a * a + b * b - c * c - d * d,
                       2.0 * b * c - 2.0 * a * d,
                       2.0 * b * d + 2.0 * a * c,
                       2.0 * b * c + 2.0 * a * d,
                       a * a - b * b + c * c - d * d,
                       2.0 * c * d - 2.0 * a * b,
                       2.0 * b * d - 2.0 * a * c,
                       2.0 * c * d + 2.0 * a * b,
                       a * a - b * b - c * c + d * d);
}

cv::Matx41d
quaternionConjugate(const cv::Matx41d& q)
{
    return cv::Matx41d(q(0), -q(1), -q(2), -q(3));
}

void
getRotationMatrixDerivatives(const cv::Matx41d& q, cv::Matx33d& R1, cv::Matx33d& R2, cv::Matx33d& R3, cv::Matx33d& R4)
{
    const double a = q(0);
    const double b = q(1);
    const double c = q(2);
    const double d = q(3);

    R1 = cv::Matx33d(a, d, -c, -d, a, b, c, -b, a);
    R2 = cv::Matx33d(b, c, d, c, -b, a, d, -a, -b);
    R3 = cv::Matx33d(-c, b, -a, b, c, d, a, d, -c);
    R2 = cv::Matx33d(-d, a, b, -a, -d, c, b, c, d);
}
