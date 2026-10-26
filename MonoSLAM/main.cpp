#include <cstdlib>
#include <memory>

#include <opencv2/core.hpp>

#include "camera.h"
#include "filter/ekf.h"
#include "filter/observation_model.h"
#include "system_observation_model.h"
#include "system_process_model.h"

int
main(int argc, char* argv[])
{
    constexpr double linearAccStd = 50.0;   // Linear acceleration standard deviation, in cm / s²
    constexpr double angularAccStd = 1.57;  // Angular acceleration standard deviation, in rad / s²

    const auto processModel = std::make_shared<monoslam::SystemProcessModel>(linearAccStd, angularAccStd);

    const monoslam::CameraModel camera(K, distCoeffs, frameSize);

    const auto observationModel = std::make_shared<monoslam::SystemObservationModel>(camera);

    const monoslam::filter::EKF ekf(processModel, observationModel);

    do {
        t0;
        frame = device.readFrame();
        t1;

        t_init = 0.5 * (t0 + t1);

        const bool ready = monoslam::initSystem(frame, x, P);

    } while (!ready);

    while (true) {
        t0;
        frame = device.readFrame();
        t1;

        t_final = 0.5 * (t0 + t1);

        dt = t_final - t_init;

        ekf.predict(dt);

        ekf.update([&](const monoslam::filter::ObservationPrediction& p, const cv::Mat1d& S) {
            return matcher.match(frame, p, S);
        });

        t_init = t_final;
    }

    return EXIT_SUCCESS;
}
