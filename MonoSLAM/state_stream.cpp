#include "state_stream.h"

#include <algorithm>
#include <cmath>

#include <opencv2/geometry.hpp>

#include "inverse_depth.h"

bool
monoslam::StateStream::open(const std::string& path)
{
    file_.open(path);

    return file_.is_open();
}

void
monoslam::StateStream::write(std::size_t frame,
                             const EKF& ekf,
                             const std::vector<Feature>& features,
                             const std::vector<std::size_t>& inView,
                             const std::vector<std::size_t>& used)
{
    const cv::Mat1d& x = ekf.state();
    const cv::Mat1d& P = ekf.covariance();

    const cv::Vec3d r = cameraPosition(x);

    cv::Vec3d o;
    cv::Rodrigues(ekf.orientation(), o);

    cv::Matx33d positionCovariance;
    P(cv::Rect(0, 0, 3, 3)).copyTo(positionCovariance);

    cv::Vec3d eigenvalues;
    cv::eigen(positionCovariance, eigenvalues);

    file_ << "frame " << frame << ' ' << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << o[0] << ' ' << o[1] << ' ' << o[2]
          << ' ' << used.size() << ' ' << std::sqrt(eigenvalues[0]) << '\n';

    for (std::size_t i = 0; i < features.size(); ++i) {
        const inverse_depth::Parameters y = featureParameters(x, i);

        const int rho = featureIndex(i) + 5;
        const double rhoStd = std::sqrt(P(rho, rho));

        // The ray within two standard deviations: the far end is at infinity if rho can be zero
        inverse_depth::Parameters nearest = y;
        inverse_depth::Parameters farthest = y;
        nearest[5] = y[5] + 2.0 * rhoStd;
        farthest[5] = y[5] - 2.0 * rhoStd;

        const cv::Vec3d p = inverse_depth::toPoint(y);
        const cv::Vec3d near = inverse_depth::toPoint(nearest);
        const cv::Vec3d far = inverse_depth::toPoint(farthest);

        int status = 0;

        if (std::find(used.begin(), used.end(), i) != used.end()) {
            status = 2;

        } else if (std::find(inView.begin(), inView.end(), i) != inView.end()) {
            status = 1;
        }

        file_ << "feature " << features[i].id << ' ' << status << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << ' '
              << near[0] << ' ' << near[1] << ' ' << near[2] << ' ' << far[0] << ' ' << far[1] << ' ' << far[2] << '\n';
    }

    file_ << "end" << std::endl;  // Flushes
}
