#ifndef MONOSLAM_MATCHER_H
#define MONOSLAM_MATCHER_H

#include <vector>

#include <opencv2/core.hpp>

#include "feature.h"
#include "filter/observation_model.h"

namespace monoslam {

class Matcher
{
  public:
    Matcher();

    filter::Measurement match(const cv::Mat1b& frame,
                              const std::vector<Feature>& features,
                              const cv::Mat1d& x,
                              const filter::ObservationPrediction& p,
                              const cv::Mat1d& S);
};

}

#endif
