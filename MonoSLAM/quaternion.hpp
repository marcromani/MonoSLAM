#ifndef MONOSLAM_QUATERNION_H
#define MONOSLAM_QUATERNION_H

#include <opencv2/core.hpp>

namespace monoslam::quaternion {

/*
 * Builds a rotation quaternion from a (non-zero) rotation vector.
 */
cv::Vec4d
fromVector(const cv::Vec3d&);

/*
 * Computes the conjugate quaternion, which is also the inverse for unit quaternions.
 */
cv::Vec4d
conjugate(const cv::Vec4d&);

/*
 * Computes the Hamilton product of two quaternions, q1 x q2.
 */
cv::Vec4d
multiply(const cv::Vec4d& q1, const cv::Vec4d& q2);

/*
 * Computes a rotation matrix from a rotation quaternion. In particular, if the
 * quaternion is an orientation quaternion, which rotates a reference frame A to
 * its final pose B, the matrix is a change of basis from B to A.
 */
cv::Matx33d
toMatrix(const cv::Vec4d&);

/*
 * Computes the matrices of the derivatives of R(q)^t with respect to the quaternion
 * components. That is, the entries of Ri are the derivatives of the entries of the
 * transpose of the matrix returned by toMatrix(q) with respect to the ith quaternion
 * component.
 */
void
toMatrixDerivatives(const cv::Vec4d& q, cv::Matx33d& R1, cv::Matx33d& R2, cv::Matx33d& R3, cv::Matx33d& R4);

}

#endif
