#include "camera.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include "util.hpp"

Camera::Camera(const cv::Matx33d& K_, const cv::Vec<double, 5>& distCoeffs_, const cv::Size& frameSize_)
  : K(K_)
  , distCoeffs(distCoeffs_)
  , frameSize(frameSize_)
{
}

Camera::Camera(const cv::Matx33d& K_, const cv::Size& frameSize_)
  : K(K_)
  , distCoeffs(cv::Vec<double, 5>::all(0.0))
  , frameSize(frameSize_)
{
}

cv::Mat_<uchar>
Camera::warpPatch(const cv::Matx31d& p,
                  const cv::Matx31d& n,
                  const cv::Mat_<uchar>& view1,
                  const cv::Rect& patch1,
                  const cv::Matx33d& R1,
                  const cv::Matx31d& t1,
                  const cv::Matx33d& R2,
                  const cv::Matx31d& t2,
                  int& u,
                  int& v,
                  cv::Mat_<uchar>& mask) const
{
    // Black background
    cv::Mat_<uchar> black = cv::Mat_<uchar>::zeros(view1.rows, view1.cols);

    // Copy the original patch into the black background
    view1(patch1).copyTo(black(patch1));

    // Compute the homography matrix between the first and second views
    const cv::Matx33d H = computeHomography(p, n, K, R1, t1, R2, t2);

    // Apply the homography
    cv::Mat_<uchar> warped;
    cv::warpPerspective(black, warped, H, view1.size());

    // Set every non-black pixel of the warped image to white
    cv::threshold(warped, mask, 0.0, 255.0, cv::THRESH_BINARY);

    // Use the binary mask to crop the warped image so that it only contains the patch
    cv::Mat_<cv::Vec2i> points;
    cv::findNonZero(mask, points);
    const cv::Rect patchBox = cv::boundingRect(points);

    // Compute the projection of the center of the patch
    const int x = patch1.x + 0.5 * patch1.width;
    const int y = patch1.y + 0.5 * patch1.height;

    cv::Matx31d centerProj = H * cv::Matx31d(x, y, 1);
    centerProj /= centerProj(2);

    u = static_cast<int>(centerProj(0) - patchBox.x);
    v = static_cast<int>(centerProj(1) - patchBox.y);

    mask = mask(patchBox);  // TODO: Should clone? Otherwise mask -> mask(patchBox) and dangling pointer?

    return warped(patchBox);
}

void
Camera::projectPoints(const cv::Matx33d& R,
                      const cv::Matx31d& t,
                      const std::vector<cv::Point3d>& points3D,
                      std::vector<cv::Point2d>& points2D) const
{
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    const double k1 = distCoeffs(0);
    const double k2 = distCoeffs(1);
    const double p1 = distCoeffs(2);
    const double p2 = distCoeffs(3);
    const double k3 = distCoeffs(4);

    points2D.resize(points3D.size());

    std::transform(points3D.begin(), points3D.end(), points2D.begin(), [&](const cv::Point3d& pt) {
        const cv::Matx31d pCam = R * (cv::Matx31d(pt) - t);

        const double zInv = 1.0 / pCam(2);
        const double xn = pCam(0) * zInv;
        const double yn = pCam(1) * zInv;

        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;

        const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        const double tangentialX = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
        const double tangentialY = 2.0 * p2 * xn * yn + p1 * (r2 + 2.0 * yn * yn);

        return cv::Point2d(fx * (xn * radial + tangentialX) + cx, fy * (yn * radial + tangentialY) + cy);
    });
}
