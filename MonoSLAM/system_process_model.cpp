#include "system_process_model.h"

monoslam::SystemProcessModel::SystemProcessModel(double linearVelStd, double angularVelStd)
{
    vel_ = cv::Matx66d::zeros();

    vel_(0, 0) = linearVelStd * linearVelStd;
    vel_(1, 1) = linearVelStd * linearVelStd;
    vel_(2, 2) = linearVelStd * linearVelStd;
    vel_(3, 3) = angularVelStd * angularVelStd;
    vel_(4, 4) = angularVelStd * angularVelStd;
    vel_(5, 5) = angularVelStd * angularVelStd;
}

monoslam::filter::ProcessPrediction
monoslam::SystemProcessModel::eval(const cv::Mat1d& x, double) const
{
    filter::ProcessPrediction p;

    p.x = x.clone();
    p.F = cv::Mat1d::eye(cameraStateSize, cameraStateSize);
    p.W = computeW(x);

    return p;
}

cv::Mat1d
monoslam::SystemProcessModel::noiseCov(double dt) const
{
    return cv::Mat1d(vel_ * dt * dt);
}

std::vector<size_t>
monoslam::SystemProcessModel::activeEntries() const
{
    return { 0, 1, 2, 3, 4, 5, 6 };
}

void
monoslam::SystemProcessModel::enforceStateConstraints(cv::Mat1d& x, cv::Mat1d& P) const
{
    cv::Vec4d q(x(3), x(4), x(5), x(6));

    const double q1 = q[0];
    const double q2 = q[1];
    const double q3 = q[2];
    const double q4 = q[3];

    const double q11 = q1 * q1;
    const double q22 = q2 * q2;
    const double q33 = q3 * q3;
    const double q44 = q4 * q4;

    const double j1 = q22 + q33 + q44;
    const double j2 = q11 + q33 + q44;
    const double j3 = q11 + q22 + q44;
    const double j4 = q11 + q22 + q33;

    const double j12 = -q1 * q2;
    const double j13 = -q1 * q3;
    const double j14 = -q1 * q4;
    const double j23 = -q2 * q3;
    const double j24 = -q2 * q4;
    const double j34 = -q3 * q4;

    cv::Matx44d J;

    J(0, 0) = j1;
    J(0, 1) = j12;
    J(0, 2) = j13;
    J(0, 3) = j14;

    J(1, 0) = j12;
    J(1, 1) = j2;
    J(1, 2) = j23;
    J(1, 3) = j24;

    J(2, 0) = j13;
    J(2, 1) = j23;
    J(2, 2) = j3;
    J(2, 3) = j34;

    J(3, 0) = j14;
    J(3, 1) = j24;
    J(3, 2) = j34;
    J(3, 3) = j4;

    const double qnorm = cv::norm(q);

    q /= qnorm;
    J /= qnorm * qnorm * qnorm;

    q.copyTo(x(cv::Rect(0, 3, 1, 4)));

    cv::Mat1d(J * P.rowRange(3, 7)).copyTo(P.rowRange(3, 7));
    cv::Mat1d(P.colRange(3, 7) * J).copyTo(P.colRange(3, 7));
}

cv::Mat1d
monoslam::SystemProcessModel::computeW(const cv::Mat1d& x) const
{
    const double q1 = x(3);
    const double q2 = x(4);
    const double q3 = x(5);
    const double q4 = x(6);

    cv::Mat1d W = cv::Mat1d::zeros(cameraStateSize, 6);

    // Derivatives of r + nr w.r.t. nr (nr = v * dt)
    W(0, 0) = 1.0;
    W(1, 1) = 1.0;
    W(2, 2) = 1.0;

    // Derivatives of q x quat(nw) w.r.t. nw (nw = w * dt)
    W(3, 3) = -0.5 * q2;
    W(3, 4) = -0.5 * q3;
    W(3, 5) = -0.5 * q4;

    W(4, 3) = 0.5 * q1;
    W(4, 4) = -0.5 * q4;
    W(4, 5) = 0.5 * q3;

    W(5, 3) = 0.5 * q4;
    W(5, 4) = 0.5 * q1;
    W(5, 5) = -0.5 * q2;

    W(6, 3) = -0.5 * q3;
    W(6, 4) = 0.5 * q2;
    W(6, 5) = 0.5 * q1;

    return W;
}
