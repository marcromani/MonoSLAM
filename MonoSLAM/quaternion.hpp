#ifndef MONOSLAM_QUATERNION_H
#define MONOSLAM_QUATERNION_H

#include <opencv2/core.hpp>

/*
 * Builds a rotation quaternion from a (non-zero) rotation vector.
 */
cv::Matx41d
computeQuaternion(const cv::Matx31d&);

/*
 * Computes the Hamilton product of two quaternions, q1 x q2.
 */
cv::Matx41d
quaternionMultiply(const cv::Matx41d& q1, const cv::Matx41d& q2);

/*
 * Computes a rotation matrix from a rotation quaternion. In particular, if the
 * quaternion is an orientation quaternion, which rotates a reference frame A to
 * its final pose B, the matrix is a change of basis from B to A.
 */
cv::Matx33d
getRotationMatrix(const cv::Matx41d&);

/*
 * Computes the conjugate quaternion, which is also the inverse for unit quaternions.
 */
cv::Matx41d
quaternionConjugate(const cv::Matx41d&);

/*
 * Computes the matrices of the derivatives of R(q)^t with respect to the quaternion
 * components. That is, the entries of Ri are the derivatives of the entries of the
 * transpose of the matrix returned by getRotationMatrix(q) with respect to the ith
 * quaternion component.
 */
void
getRotationMatrixDerivatives(const cv::Matx41d& q, cv::Matx33d& R1, cv::Matx33d& R2, cv::Matx33d& R3, cv::Matx33d& R4);

#endif
