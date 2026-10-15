#ifndef MONOSLAM_SYSTEM_OBSERVATION_MODEL_H
#define MONOSLAM_SYSTEM_OBSERVATION_MODEL_H

#include <opencv2/core.hpp>

#include "camera.h"
#include "filter/observation_model.h"

namespace monoslam {

class SystemObservationModel : public filter::ObservationModel
{
  public:
    SystemObservationModel(const CameraModel& cameraModel, double pixelStd);

    filter::Prediction eval(const cv::Mat1d& x) const override;

  private:
    CameraModel cameraModel_;  // Camera model, provides projections and their Jacobians
    double pixelStd_;          // Standard deviation of the noise in a pixel measurement
};

}

#endif
