#ifndef MONOSLAM_SYSTEM_OBSERVATION_MODEL_H
#define MONOSLAM_SYSTEM_OBSERVATION_MODEL_H

#include <opencv2/core.hpp>

#include "camera.h"
#include "filter/observation_model.h"

namespace monoslam {

class SystemObservationModel : public filter::ObservationModel
{
  public:
    SystemObservationModel(const CameraModel& cameraModel);

    filter::Prediction eval(const cv::Mat1d& x) const override;

  private:
    cv::Mat1d h(const cv::Mat1d& x) const;
    cv::Mat1d computeH(const cv::Mat1d& x) const;

    CameraModel cameraModel_;
};

}

#endif
