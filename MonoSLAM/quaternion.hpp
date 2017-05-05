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

/*
 * Computes the matrices of the derivatives of R(q)^t with respect to the quaternion
 * components. That is, the entries of Ri are the derivatives of the entries of the
 * transpose of the matrix returned by getRotationMatrix(q) with respect to the ith
 * quaternion component.
 */
void getRotationMatrixDerivatives(const cv::Mat& q, cv::Mat& R1, cv::Mat& R2, cv::Mat& R3, cv::Mat& R4);

#endif