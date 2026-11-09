#ifndef MONOSLAM_INVERSE_DEPTH_H
#define MONOSLAM_INVERSE_DEPTH_H

#include <opencv2/core.hpp>

#include "camera.h"

namespace monoslam::inverse_depth {

/*
 * Inverse depth parametrization of a point (Civera, Davison and Montiel, 2008). A point is represented by six
 * parameters y = (a, theta, phi, rho). The anchor a is the position of the camera when the point was first seen. theta
 * and phi are the azimuth and elevation of the ray from the anchor to the point, in world coordinates, and rho
 * is the inverse of the distance along that ray. The point is
 *
 *     p = a + m(theta, phi) / rho,    m = (cos(phi) sin(theta), -sin(phi), cos(phi) cos(theta)).
 *
 * Unlike the depth, the inverse depth has a nearly Gaussian uncertainty from the first time a point is seen, even
 * if the point is far away or at infinity (rho = 0). A point can therefore be added to the filter right away,
 * and constrain the orientation of the camera before there is enough parallax to know its depth.
 */

constexpr int size = 6;

using Parameters = cv::Vec<double, size>;

/*
 * Returns the unit vector m(theta, phi).
 */
cv::Vec3d
direction(double theta, double phi);

/*
 * Returns the azimuth and elevation of a direction, which does not need to be a unit vector, and the Jacobian
 * of (theta, phi) with respect to it.
 *
 * d    Direction
 * J    Jacobian of (theta, phi) with respect to d
 */
cv::Vec2d
angles(const cv::Vec3d& d, cv::Matx23d& J);

/*
 * Returns the parameters of a point seen from an anchor.
 *
 * anchor    Anchor
 * point     Point, in world coordinates
 */
Parameters
fromPoint(const cv::Vec3d& anchor, const cv::Vec3d& point);

/*
 * Returns the Jacobian of (theta, phi, rho) with respect to the point, for a fixed anchor.
 *
 * anchor    Anchor
 * point     Point, in world coordinates
 */
cv::Matx33d
fromPointJacobian(const cv::Vec3d& anchor, const cv::Vec3d& point);

/*
 * Returns the point in world coordinates. Points too far away to have a meaningful position, rho <= 0.001, are placed
 * at distance 1000 along their ray, which is what the patch warping needs.
 *
 * y    Parameters
 */
cv::Vec3d
toPoint(const Parameters& y);

/*
 * Returns the parameters of a point first seen at a pixel, with a given inverse depth, and the Jacobian
 * of the parameters with respect to the camera position, the camera orientation, the pixel and the inverse depth,
 * in this order (6 x 9). The orientation is that of the filter, a small rotation e of the camera, R exp(e), see ekf.h.
 *
 * pose            Camera pose
 * camera          Camera model
 * pixel           Pixel where the point is seen
 * inverseDepth    Initial inverse depth
 * J               Jacobian of the parameters
 */
Parameters
fromPixel(const CameraPose& pose,
          const CameraModel& camera,
          const cv::Point2d& pixel,
          double inverseDepth,
          cv::Matx<double, size, 9>& J);

/*
 * Returns the point in camera coordinates, scaled by rho, h = R^T (rho (a - r) + m), and its Jacobians. The scale does
 * not change the pixel, and it keeps h well defined for points at infinity.
 *
 * pose    Camera pose
 * y       Parameters
 * Jr      Jacobian of h with respect to the camera position
 * Je      Jacobian of h with respect to a small rotation e of the camera, R exp(e)
 * Jy      Jacobian of h with respect to the parameters
 */
cv::Vec3d
toCamera(const CameraPose& pose, const Parameters& y, cv::Matx33d& Jr, cv::Matx33d& Je, cv::Matx<double, 3, size>& Jy);

}

#endif
