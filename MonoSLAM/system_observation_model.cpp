#include "system_observation_model.h"

#include <vector>

monoslam::SystemObservationModel::SystemObservationModel(const CameraModel& cameraModel)
  : cameraModel_(cameraModel)
{
}

monoslam::filter::Prediction
monoslam::SystemObservationModel::eval(const cv::Mat1d& x) const
{
    const int numFeatures = (x.rows - 13) / 3;

    const cv::Vec3d r(x(0), x(1), x(2));
    const cv::Vec4d q(x(3), x(4), x(5), x(6));

    const CameraPose pose(r, q);

    std::vector<Projection> projections;
    std::vector<int> visible;

    projections.reserve(numFeatures);
    visible.reserve(numFeatures);

    for (int i = 0; i < numFeatures; ++i) {
        const int idx = 13 + 3 * i;

        const cv::Vec3d p(x(idx), x(idx + 1), x(idx + 2));

        const std::optional<Projection> result = cameraModel_.projectWithJacobian(pose, p);

        if (result.has_value()) {
            projections.push_back(result.value());
            visible.push_back(idx);
        }
    }

    // TODO: Continue here
}
