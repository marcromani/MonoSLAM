#include "system_process_model.h"

#include "quaternion.h"

monoslam::SystemProcessModel::SystemProcessModel(double linearAccStd, double angularAccStd)
{
    acc_ = cv::Matx66d::zeros();

    acc_(0, 0) = linearAccStd * linearAccStd;
    acc_(1, 1) = linearAccStd * linearAccStd;
    acc_(2, 2) = linearAccStd * linearAccStd;
    acc_(3, 3) = angularAccStd * angularAccStd;
    acc_(4, 4) = angularAccStd * angularAccStd;
    acc_(5, 5) = angularAccStd * angularAccStd;
}

monoslam::filter::ProcessPrediction
monoslam::SystemProcessModel::eval(const cv::Mat1d& x, double dt) const
{
    filter::ProcessPrediction p;

    p.x = f(x, dt);
    p.F = computeF(x, dt);
    p.W = computeW(dt, p.F);

    return p;
}

cv::Mat1d
monoslam::SystemProcessModel::noiseCov(double dt) const
{
    return cv::Mat1d(acc_ * dt * dt);
}

std::vector<size_t>
monoslam::SystemProcessModel::activeEntries() const
{
    return { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
}

cv::Mat1d
monoslam::SystemProcessModel::f(const cv::Mat1d& x, double dt) const
{
    cv::Mat1d xPred = x.clone();

    xPred(cv::Rect(0, 0, 1, 3)) += xPred(cv::Rect(0, 7, 1, 3)) * dt;

    if (xPred(10) != 0.0 || xPred(11) != 0.0 || xPred(12) != 0.0) {
        const cv::Vec3d w(xPred(10), xPred(11), xPred(12));

        const cv::Vec4d q = quaternion::multiply(xPred(cv::Rect(0, 3, 1, 4)), quaternion::fromVector(w * dt));

        for (int i = 0; i < 4; ++i) {
            xPred(3 + i) = q[i];
        }
    }

    return xPred;
}

cv::Mat1d
monoslam::SystemProcessModel::computeF(const cv::Mat1d& x, double dt) const
{
    cv::Mat1d F = cv::Mat1d::eye(13, 13);

    // Set J{r_new / v_old}

    F(0, 7) = dt;
    F(1, 8) = dt;
    F(2, 9) = dt;

    // Compute J{q_new / q(w*dt)}

    cv::Matx44d Q1;

    // Current pose quaternion (q_old)
    const double a1 = x(3);
    const double b1 = x(4);
    const double c1 = x(5);
    const double d1 = x(6);

    // Row 1
    Q1(0, 0) = a1;
    Q1(0, 1) = -b1;
    Q1(0, 2) = -c1;
    Q1(0, 3) = -d1;

    // Row 2
    Q1(1, 0) = b1;
    Q1(1, 1) = a1;
    Q1(1, 2) = -d1;
    Q1(1, 3) = c1;

    // Row 3
    Q1(2, 0) = c1;
    Q1(2, 1) = d1;
    Q1(2, 2) = a1;
    Q1(2, 3) = -b1;

    // Row 4
    Q1(3, 0) = d1;
    Q1(3, 1) = -c1;
    Q1(3, 2) = b1;
    Q1(3, 3) = a1;

    cv::Matx43d Q2;

    const double w1 = x(10);
    const double w2 = x(11);
    const double w3 = x(12);

    if (w1 == 0.0 && w2 == 0.0 && w3 == 0.0) {
        // Compute J{q(w*dt) / w_old}

        Q2 = cv::Matx43d::zeros();

        Q2(1, 0) = 0.5 * dt;
        Q2(2, 1) = 0.5 * dt;
        Q2(3, 2) = 0.5 * dt;

    } else {
        // Set J{q_new / q_old}

        const cv::Vec3d rotationVector(x(10), x(11), x(12));

        // Rotation quaternion q(w*dt)
        const cv::Vec4d q = quaternion::fromVector(rotationVector * dt);

        const double a2 = q[0];
        const double b2 = q[1];
        const double c2 = q[2];
        const double d2 = q[3];

        // Row 1
        F(3, 3) = a2;
        F(3, 4) = -b2;
        F(3, 5) = -c2;
        F(3, 6) = -d2;

        // Row 2
        F(4, 3) = b2;
        F(4, 4) = a2;
        F(4, 5) = d2;
        F(4, 6) = -c2;

        // Row 3
        F(5, 3) = c2;
        F(5, 4) = -d2;
        F(5, 5) = a2;
        F(5, 6) = b2;

        // Row 4
        F(6, 3) = d2;
        F(6, 4) = c2;
        F(6, 5) = -b2;
        F(6, 6) = a2;

        // Compute J{q(w*dt) / w_old}

        const double w11 = w1 * w1;
        const double w22 = w2 * w2;
        const double w33 = w3 * w3;

        const double norm2 = w11 + w22 + w33;
        const double norm = std::sqrt(norm2);

        const double c = std::cos(0.5 * dt * norm);
        const double s = std::sin(0.5 * dt * norm);

        const double z = s / norm;

        const double dq1dw_ = -0.5 * dt * z;
        const double dq_dw_ = 0.5 * dt * c / norm2 - z / norm2;

        const double w12 = w1 * w2;
        const double w13 = w1 * w3;
        const double w23 = w2 * w3;

        // Row 1
        Q2(0, 0) = dq1dw_ * w1;
        Q2(0, 1) = dq1dw_ * w2;
        Q2(0, 2) = dq1dw_ * w3;

        // Row 2
        Q2(1, 0) = dq_dw_ * w11 + z;
        Q2(1, 1) = dq_dw_ * w12;
        Q2(1, 2) = dq_dw_ * w13;

        // Row 3
        Q2(2, 0) = dq_dw_ * w12;
        Q2(2, 1) = dq_dw_ * w22 + z;
        Q2(2, 2) = dq_dw_ * w23;

        // Row 4
        Q2(3, 0) = dq_dw_ * w13;
        Q2(3, 1) = dq_dw_ * w23;
        Q2(3, 2) = dq_dw_ * w33 + z;
    }

    const cv::Matx43d M = Q1 * Q2;

    // Set J{q_new / w_old}

    F(cv::Rect(10, 3, 3, 4)) = cv::Mat1d(M, false);

    return F;
}

cv::Mat1d
monoslam::SystemProcessModel::computeW(double dt, const cv::Mat1d& F) const
{
    cv::Matx<double, 13, 6> W = cv::Matx<double, 13, 6>::zeros();

    // Set J{r_new / big_v}

    W(0, 0) = dt;
    W(1, 1) = dt;
    W(2, 2) = dt;

    // Set J{q_new / big_w}

    for (int i = 3; i < 7; ++i) {
        for (int j = 3; j < 6; ++j) {
            W(i, j) = F(i, 7 + j);
        }
    }

    // Set the bottom identity matrix, J{v_new / big_v} and J{w_new / big_w}

    for (int i = 7; i < 13; ++i) {
        W(i, i) = 1.0;
    }

    return cv::Mat1d(W);
}
