#include "inverse_depth.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double minInverseDepth = 1e-3;  // Points further away than 1 / minInverseDepth are at infinity

}

cv::Vec3d
monoslam::inverse_depth::direction(double theta, double phi)
{
    return cv::Vec3d(std::cos(phi) * std::sin(theta), -std::sin(phi), std::cos(phi) * std::cos(theta));
}

cv::Vec2d
monoslam::inverse_depth::angles(const cv::Vec3d& d, cv::Matx23d& J)
{
    const double horizontal2 = d[0] * d[0] + d[2] * d[2];  // Squared length of the projection on the xz plane
    const double horizontal = std::sqrt(horizontal2);
    const double length2 = horizontal2 + d[1] * d[1];

    // theta = atan2(dx, dz), phi = atan2(-dy, horizontal)
    J = cv::Matx23d(d[2] / horizontal2,
                    0.0,
                    -d[0] / horizontal2,
                    d[1] * d[0] / (horizontal * length2),
                    -horizontal / length2,
                    d[1] * d[2] / (horizontal * length2));

    return cv::Vec2d(std::atan2(d[0], d[2]), std::atan2(-d[1], horizontal));
}

monoslam::inverse_depth::Parameters
monoslam::inverse_depth::fromPoint(const cv::Vec3d& anchor, const cv::Vec3d& point)
{
    const cv::Vec3d d = point - anchor;

    cv::Matx23d J;
    const cv::Vec2d thetaPhi = angles(d, J);

    return Parameters(anchor[0], anchor[1], anchor[2], thetaPhi[0], thetaPhi[1], 1.0 / cv::norm(d));
}

cv::Matx33d
monoslam::inverse_depth::fromPointJacobian(const cv::Vec3d& anchor, const cv::Vec3d& point)
{
    const cv::Vec3d d = point - anchor;
    const double length = cv::norm(d);

    cv::Matx23d Jangles;
    angles(d, Jangles);

    // rho = 1 / |d|, so its gradient is -d / |d|^3
    const cv::Vec3d Jrho = -d / (length * length * length);

    return cv::Matx33d(Jangles(0, 0),
                       Jangles(0, 1),
                       Jangles(0, 2),
                       Jangles(1, 0),
                       Jangles(1, 1),
                       Jangles(1, 2),
                       Jrho[0],
                       Jrho[1],
                       Jrho[2]);
}

cv::Vec3d
monoslam::inverse_depth::toPoint(const Parameters& y)
{
    const cv::Vec3d anchor(y[0], y[1], y[2]);

    return anchor + direction(y[3], y[4]) / std::max(y[5], minInverseDepth);
}

monoslam::inverse_depth::Parameters
monoslam::inverse_depth::fromPixel(const CameraPose& pose,
                                   const CameraModel& camera,
                                   const cv::Point2d& pixel,
                                   double inverseDepth,
                                   cv::Matx<double, size, 9>& J)
{
    // Ray of the pixel in camera coordinates, and its derivative with respect to the pixel, which is the inverse
    // of the derivative of the projection
    const cv::Vec3d rayCamera = camera.unproject(pixel);
    const cv::Matx22d dNormalized = camera.distortionJacobian(rayCamera[0], rayCamera[1]).inv();

    // Ray in world coordinates
    const cv::Vec3d ray = pose.R() * rayCamera;

    cv::Matx23d Jangles;
    const cv::Vec2d thetaPhi = angles(ray, Jangles);

    J = cv::Matx<double, size, 9>::zeros();

    // The anchor is the camera position
    for (int i = 0; i < 3; ++i) {
        J(i, i) = 1.0;
    }

    // The angles depend on the orientation, through the ray in world coordinates: rotating the camera by e turns
    // the ray into R exp(e) ray_c ~ R (ray_c + e x ray_c) = ray - R [ray_c]x e
    const cv::Matx23d dAnglesOrientation = Jangles * (-pose.R() * skew(rayCamera));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            J(3 + i, 3 + j) = dAnglesOrientation(i, j);
        }
    }

    // And on the pixel, through the ray in camera coordinates (x, y, 1)
    const cv::Matx32d dRayCamera(dNormalized(0, 0), dNormalized(0, 1), dNormalized(1, 0), dNormalized(1, 1), 0.0, 0.0);
    const cv::Matx22d dAnglesPixel = Jangles * pose.R() * dRayCamera;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            J(3 + i, 6 + j) = dAnglesPixel(i, j);
        }
    }

    // The inverse depth is a parameter of its own
    J(5, 8) = 1.0;

    const cv::Vec3d anchor = pose.r();

    return Parameters(anchor[0], anchor[1], anchor[2], thetaPhi[0], thetaPhi[1], inverseDepth);
}

cv::Vec3d
monoslam::inverse_depth::toCamera(const CameraPose& pose,
                                  const Parameters& y,
                                  cv::Matx33d& Jr,
                                  cv::Matx33d& Je,
                                  cv::Matx<double, 3, size>& Jy)
{
    const double theta = y[3];
    const double phi = y[4];
    const double rho = y[5];

    const cv::Vec3d anchor(y[0], y[1], y[2]);
    const cv::Vec3d m = direction(theta, phi);

    const cv::Vec3d hWorld = rho * (anchor - pose.r()) + m;  // h in world coordinates
    const cv::Matx33d& Rcw = pose.Rcw();

    const cv::Vec3d h = Rcw * hWorld;

    Jr = -rho * Rcw;

    // Rotating the camera by e turns h into exp(-e) h ~ h - e x h = h + [h]x e
    Je = skew(h);

    const cv::Vec3d dmdtheta(std::cos(phi) * std::cos(theta), 0.0, -std::cos(phi) * std::sin(theta));
    const cv::Vec3d dmdphi(-std::sin(phi) * std::sin(theta), -std::cos(phi), -std::sin(phi) * std::cos(theta));

    const cv::Matx33d Janchor = rho * Rcw;
    const cv::Vec3d Jtheta = Rcw * dmdtheta;
    const cv::Vec3d Jphi = Rcw * dmdphi;
    const cv::Vec3d Jrho = Rcw * (anchor - pose.r());

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Jy(i, j) = Janchor(i, j);
        }

        Jy(i, 3) = Jtheta[i];
        Jy(i, 4) = Jphi[i];
        Jy(i, 5) = Jrho[i];
    }

    return h;
}
