#include "geometry.h"

#include <vector>

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

cv::Matx33d
monoslam::geometry::computeHomography(const cv::Vec3d& p,
                                      const cv::Vec3d& n,
                                      const cv::Matx33d& K,
                                      const cv::Matx33d& R1,
                                      const cv::Vec3d& t1,
                                      const cv::Matx33d& R2,
                                      const cv::Vec3d& t2)
{
    const cv::Matx<double, 1, 1> m = n.t() * (p - t1);
    const cv::Matx33d H = K * R2 * (m(0) * cv::Matx33d::eye() + (t1 - t2) * n.t()) * R1.t() * K.inv();

    return H;
}

cv::Mat1b
monoslam::geometry::warpPatch(const cv::Vec3d& p,
                              const cv::Vec3d& n,
                              const cv::Mat1b& view1,
                              const cv::Rect& patch1,
                              const cv::Matx33d& K,
                              const cv::Matx33d& R1,
                              const cv::Vec3d& t1,
                              const cv::Matx33d& R2,
                              const cv::Vec3d& t2,
                              double& u,
                              double& v,
                              cv::Mat1b& mask)
{
    assert(0 <= patch1.x &&
           0 <= patch1.y &&
           patch1.x + patch1.width <= view1.cols &&
           patch1.y + patch1.height <= view1.rows);

    // Compute the homography matrix between the first and second views
    const cv::Matx33d H = computeHomography(p, n, K, R1, t1, R2, t2);

    // Apply the homography
    cv::Mat1b warped;
    cv::warpPerspective(view1, warped, H, view1.size());

    // Warp a solid white copy of the patch to build a shape mask
    cv::Mat1b shape = cv::Mat1b::zeros(view1.size());
    shape(patch1).setTo(255);

    // Warp the shape mask
    cv::warpPerspective(shape, mask, H, view1.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0.0));

    // Compute the bounding box of the warped patch using the warped shape mask
    std::vector<cv::Point2i> points;
    cv::findNonZero(mask, points);

    if (points.empty()) {
        mask = {};
        return {};
    }

    const cv::Rect patchBox = cv::boundingRect(points);

    // Map the center of the patch to the new view
    const double x = patch1.x + 0.5 * patch1.width;
    const double y = patch1.y + 0.5 * patch1.height;

    const cv::Vec3d center = H * cv::Vec3d(x, y, 1);

    u = center[0] / center[2] - patchBox.x;
    v = center[1] / center[2] - patchBox.y;

    mask = mask(patchBox);

    return warped(patchBox);
}
