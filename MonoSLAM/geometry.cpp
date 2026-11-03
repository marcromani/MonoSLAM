#include "geometry.h"

#include <vector>

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

bool
monoslam::geometry::computeHomography(const cv::Vec3d& p,
                                      const cv::Vec3d& n,
                                      const cv::Rect& region,
                                      const CameraModel& camera,
                                      const cv::Matx33d& R1,
                                      const cv::Vec3d& t1,
                                      const cv::Matx33d& R2,
                                      const cv::Vec3d& t2,
                                      cv::Matx33d& H)
{
    const cv::Point2f topLeft(region.tl());
    const cv::Point2f bottomRight(region.br());

    const std::vector<cv::Point2f> corners1 = {
        topLeft, cv::Point2f(bottomRight.x, topLeft.y), bottomRight, cv::Point2f(topLeft.x, bottomRight.y)
    };

    const double m = n.dot(p - t1);

    std::vector<cv::Point2f> corners2;

    for (const cv::Point2f& corner : corners1) {
        const cv::Vec3d direction = R1.t() * camera.unproject(corner);  // Ray through the corner, in world coordinates
        const cv::Vec3d point = t1 + direction * (m / n.dot(direction));  // Intersection of the ray with the plane
        const cv::Vec3d pointCamera = R2 * (point - t2);

        if (pointCamera[2] <= 0.0) {
            return false;
        }

        corners2.emplace_back(camera.distort(pointCamera[0] / pointCamera[2], pointCamera[1] / pointCamera[2]));
    }

    H = cv::Matx33d(cv::getPerspectiveTransform(corners1, corners2));

    return true;
}

cv::Mat1b
monoslam::geometry::warpPatch(const cv::Vec3d& p,
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
                              cv::Mat1b& mask)
{
    assert(0 <= patch1.x &&
           0 <= patch1.y &&
           patch1.x + patch1.width <= view1.cols &&
           patch1.y + patch1.height <= view1.rows);

    // Compute the homography matrix between the first and second views
    cv::Matx33d H;

    if (!computeHomography(p, n, patch1, camera, R1, t1, R2, t2, H)) {
        mask = {};
        return {};
    }

    // Compute the bounding box of the warped patch to avoid warping the whole view
    const cv::Point2f topLeft(patch1.tl());
    const cv::Point2f bottomRight(patch1.br());

    const std::vector<cv::Point2f> corners = {
        topLeft, cv::Point2f(bottomRight.x, topLeft.y), bottomRight, cv::Point2f(topLeft.x, bottomRight.y)
    };

    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(corners, warpedCorners, cv::Matx33f(H));

    const cv::Rect patchBox = cv::boundingRect(warpedCorners) & cv::Rect(0, 0, view1.cols, view1.rows);

    if (patchBox.empty()) {
        mask = {};
        return {};
    }

    // Apply the homography, with the origin of the output at the corner of the box
    const cv::Matx33d toBox(1.0, 0.0, -patchBox.x, 0.0, 1.0, -patchBox.y, 0.0, 0.0, 1.0);

    cv::Mat1b warped;
    cv::warpPerspective(view1, warped, toBox * H, patchBox.size());

    // Warp a solid white copy of the patch to build a shape mask
    cv::Mat1b shape(view1.size(), 0);
    shape(patch1).setTo(255);

    cv::warpPerspective(shape, mask, toBox * H, patchBox.size());

    if (cv::countNonZero(mask) == 0) {
        mask = {};
        return {};
    }

    // Map the center of the patch to the new view
    const cv::Vec3d center = H * cv::Vec3d(anchor1.x, anchor1.y, 1);

    anchor2.x = center[0] / center[2] - patchBox.x;
    anchor2.y = center[1] / center[2] - patchBox.y;

    return warped;
}
