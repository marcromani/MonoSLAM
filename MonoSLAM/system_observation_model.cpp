#include "system_observation_model.h"

#include <numeric>
#include <vector>

monoslam::SystemObservationModel::SystemObservationModel(const CameraModel& cameraModel, double pixelStd)
  : cameraModel_(cameraModel)
  , pixelStd_(pixelStd)
{
}

monoslam::filter::ObservationPrediction
monoslam::SystemObservationModel::eval(const cv::Mat1d& x) const
{
    const size_t numFeatures = static_cast<size_t>((x.rows - 13) / 3);

    const cv::Vec3d r(x(0), x(1), x(2));
    const cv::Vec4d q(x(3), x(4), x(5), x(6));

    const CameraPose pose(r, q);

    std::vector<Projection> projections;
    std::vector<int> visible;

    projections.reserve(numFeatures);
    visible.reserve(numFeatures);

    for (size_t i = 0; i < numFeatures; ++i) {
        const int idx = static_cast<int>(13 + 3 * i);

        const cv::Vec3d point(x(idx), x(idx + 1), x(idx + 2));

        const std::optional<Projection> result = cameraModel_.projectWithJacobian(pose, point);

        if (result.has_value()) {
            projections.push_back(result.value());
            visible.push_back(idx);
        }
    }

    const int m = static_cast<int>(visible.size());

    filter::ObservationPrediction p;

    p.h = cv::Mat1d(2 * m, 1);
    p.H = cv::Mat1d(2 * m, 7 + 3 * m);
    p.R = pixelStd_ * pixelStd_ * cv::Mat1d::eye(2 * m, 2 * m);

    // There is a column in H for each component of a visible feature, plus r and q
    p.active.resize(7 + 3 * m);

    // Set the indices for r and q
    std::iota(p.active.begin(), p.active.begin() + 7, 0);

    for (int i = 0; i < m; ++i) {
        const Projection& proj = projections[i];

        p.h(i) = proj.pixel[0];
        p.h(i) = proj.pixel[1];

        proj.Jr.copyTo(p.H(cv::Rect(0, 2 * i, 3, 2)));
        proj.Jq.copyTo(p.H(cv::Rect(3, 2 * i, 4, 2)));
        proj.Jp.copyTo(p.H(cv::Rect(7 + 3 * i, 2 * i, 3, 2)));

        // Set the indices for the ith feature
        p.active[7 + 3 * i] = visible[i];
        p.active[8 + 3 * i] = visible[i + 1];
        p.active[9 + 3 * i] = visible[i + 2];
    }

    return p;
}
