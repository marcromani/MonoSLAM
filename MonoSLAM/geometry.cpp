#include "geometry.h"

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
                              int& u,
                              int& v,
                              cv::Mat1b& mask)
{
    // Black background
    cv::Mat1b black = cv::Mat1b::zeros(view1.rows, view1.cols);

    // Copy the original patch into the black background
    view1(patch1).copyTo(black(patch1));

    // Compute the homography matrix between the first and second views
    const cv::Matx33d H = computeHomography(p, n, K, R1, t1, R2, t2);

    // Apply the homography
    cv::Mat1b warped;
    cv::warpPerspective(black, warped, H, view1.size());

    // Set every non-black pixel of the warped image to white
    cv::threshold(warped, mask, 0.0, 255.0, cv::THRESH_BINARY);

    // Use the binary mask to crop the warped image so that it only contains the patch
    cv::Mat2i points;
    cv::findNonZero(mask, points);
    const cv::Rect patchBox = cv::boundingRect(points);

    // Compute the projection of the center of the patch
    const int x = static_cast<int>(patch1.x + 0.5 * patch1.width);
    const int y = static_cast<int>(patch1.y + 0.5 * patch1.height);

    cv::Vec3d centerProj = H * cv::Vec3d(x, y, 1);
    centerProj /= centerProj[2];

    u = static_cast<int>(centerProj[0] - patchBox.x);
    v = static_cast<int>(centerProj[1] - patchBox.y);

    mask = mask(patchBox);  // TODO: Should clone? Otherwise mask -> mask(patchBox) and dangling pointer?

    return warped(patchBox);
}
