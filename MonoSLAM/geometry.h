#ifndef MONOSLAM_GEOMETRY_H
#define MONOSLAM_GEOMETRY_H

#include <opencv2/core.hpp>

#include "camera.h"

namespace monoslam::geometry {

/*
 * Computes the homography that maps the corners of a rectangular region of the first view
 * to the second one, given that they are the projection of points of a plane in space. In
 * particular, p2 = H * p1, where p1 is a pixel in the first view and p2 is the pixel which
 * corresponds to the same point in the second view. A true homography only exists for a
 * pinhole camera, so the corners are mapped through the whole camera model, and the result
 * is the best approximation to the actual map within the region. Returns false if the
 * plane is seen from behind in one of the views.
 *
 * p         Point in the plane, in world coordinates
 * n         Plane normal, in world coordinates
 * region    Region of the first view
 * camera    Camera model
 * R1        First pose rotation (world to camera)
 * t1        First pose translation, in world coordinates
 * R2        Second pose rotation (world to camera)
 * t2        Second pose translation, in world coordinates
 * H         Homography matrix
 */
bool
computeHomography(const cv::Vec3d& p,
                  const cv::Vec3d& n,
                  const cv::Rect& region,
                  const CameraModel& camera,
                  const cv::Matx33d& R1,
                  const cv::Vec3d& t1,
                  const cv::Matx33d& R2,
                  const cv::Vec3d& t2,
                  cv::Matx33d& H);

/*
 * Warps a rectangular patch (presumed to be the projection of a certain
 * region of a plane in space) from one camera view to another. In particular,
 * a map M is computed such that s2 = M * s1, where s1 is the patch in the
 * first view and s2 is the same patch as would be seen from the second view.
 *
 * p          Point in the plane, in world coordinates
 * n          Plane normal, in world coordinates
 * view1      Original view
 * anchor1    Center of the patch
 * patch1     Original patch
 * camera     Camera model
 * R1         First pose rotation (world to camera)
 * t1         First pose translation, in world coordinates
 * R2         Second pose rotation (world to camera)
 * t2         Second pose translation, in world coordinates
 * anchor2    Projection of the original patch center, relative to the warped patch
 * mask       Warped patch mask, it specifies the pixels that belong to the original patch
 */
cv::Mat1b
warpPatch(const cv::Vec3d& p,
          const cv::Vec3d& n,
          const cv::Mat1b& view1,
          const cv::Point2d& anchor1,
          const cv::Rect& patch1,
          const CameraModel& camera,
          const cv::Matx33d& R1,
          const cv::Vec3d& t1,
          const cv::Matx33d& R2,
          const cv::Vec3d& t2,
          cv::Point2d& anchor2,
          cv::Mat1b& mask);

}

#endif
