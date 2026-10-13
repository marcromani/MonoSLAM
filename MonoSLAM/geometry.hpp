#ifndef MONOSLAM_GEOMETRY_H
#define MONOSLAM_GEOMETRY_H

#include <opencv2/core.hpp>

namespace monoslam::geometry {

/*
 * Computes the homography matrix H between two pinhole camera views of a plane.
 * In particular, p2 = H * p1, where p1 is the pixel in the first view which
 * corresponds to a point p in the plane, and p2 is the pixel which corresponds
 * to p in the second view.
 *
 * p     Point in the plane, in world coordinates
 * n     Plane normal, in world coordinates
 * K     Camera intrinsic matrix
 * R1    First pose rotation (world to camera)
 * t1    First pose translation, in world coordinates
 * R2    Second pose rotation (world to camera)
 * t2    Second pose translation, in world coordinates
 */
cv::Matx33d
computeHomography(const cv::Vec3d& p,
                  const cv::Vec3d& n,
                  const cv::Matx33d& K,
                  const cv::Matx33d& R1,
                  const cv::Vec3d& t1,
                  const cv::Matx33d& R2,
                  const cv::Vec3d& t2);

/*
 * Warps a rectangular patch (presumed to be the projection of a certain
 * region of a plane in space) from one camera view to another. In particular,
 * a map M is computed such that s2 = M * s1, where s1 is the patch in the
 * first view and s2 is the same patch as would be seen from the second view.
 * A pinhole camera model is assumed (i.e. zero distortion coefficients) so that
 * the computed map be a planar homography. This remains a good approximation
 * provided the planar patch is small or the lens distortion is low.
 *
 * p         Point in the plane, in world coordinates
 * n         Plane normal, in world coordinates
 * view1     Original view
 * patch1    Original patch
 * R1        First pose rotation (world to camera)
 * t1        First pose translation, in world coordinates
 * R2        Second pose rotation (world to camera)
 * t2        Second pose translation, in world coordinates
 * u         First pixel coordinate of the projection of the original patch center
 * v         Second pixel coordinate of the projection of the original patch center
 * mask      Warped patch mask, it specifies the pixels that belong to the original patch
 */
cv::Mat1b
warpPatch(const cv::Vec3d& p,
          const cv::Vec3d& n,
          const cv::Mat1b& view1,
          const cv::Rect& patch1,
          const cv::Matx33d& R1,
          const cv::Vec3d& t1,
          const cv::Matx33d& R2,
          const cv::Vec3d& t2,
          int& u,
          int& v,
          cv::Mat1b& mask);

}

#endif
