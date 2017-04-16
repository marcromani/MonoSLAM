#ifndef QUATERNION_H
#define QUATERNION_H

#include "opencv2/core/core.hpp"

/*
 * Builds a rotation quaternion from a (non-zero) rotation vector.
 */
cv::Mat computeQuaternion(const cv::Mat&);

/*
 * Computes the Hamilton product of two quaternions.
 */
cv::Mat quaternionMultiply(const cv::Mat&, const cv::Mat&);

/*
 * Computes a rotation matrix from a rotation quaternion. In particular, if the
 * quaternion is an orientation quaternion, which rotates a reference frame A to
 * its final pose B, the matrix is a change of basis from B to A.
 */
cv::Mat getRotationMatrix(const cv::Mat&);

/*
 * Computes the inverse quaternion.
 */
cv::Mat quaternionInv(const cv::Mat&);

#endif