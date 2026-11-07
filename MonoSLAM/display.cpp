#include "display.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

std::vector<monoslam::DrawnMatch>
monoslam::usedMatches(const Observation& o,
                      const std::vector<cv::Point2f>& seeds,
                      const std::vector<std::optional<cv::Point2d>>& matches,
                      const std::vector<std::size_t>& used)
{
    constexpr double numSigmas = 3.035;  // The 99% region of a 2D Gaussian

    std::vector<DrawnMatch> drawn;

    for (std::size_t k = 0; k < o.features.size(); ++k) {
        const std::size_t feature = o.features[k];

        if (!matches[k].has_value() || std::find(used.begin(), used.end(), feature) == used.end()) {
            continue;
        }

        const int row = static_cast<int>(2 * k);
        const cv::Point2d prediction(o.h(row), o.h(row + 1));

        // The axes of the ellipse are the eigenvectors of the innovation covariance
        cv::Vec2d eigenvalues;
        cv::Matx22d eigenvectors;
        cv::eigen(cv::Matx22d(o.S(cv::Rect(row, row, 2, 2))), eigenvalues, eigenvectors);

        const cv::Size2d size(2.0 * numSigmas * std::sqrt(eigenvalues[0]), 2.0 * numSigmas * std::sqrt(eigenvalues[1]));
        const double angle = std::atan2(eigenvectors(0, 1), eigenvectors(0, 0)) * 180.0 / CV_PI;

        drawn.push_back({ cv::RotatedRect(prediction, size, static_cast<float>(angle)),
                          prediction,
                          seeds[feature],
                          matches[k].value() });
    }

    return drawn;
}

cv::Mat3b
monoslam::draw(const Snapshot& snapshot)
{
    const cv::Scalar cyan(230.0, 200.0, 60.0);  // BGR
    const cv::Scalar amber(0.0, 170.0, 255.0);
    const cv::Scalar magenta(200.0, 60.0, 230.0);
    const cv::Scalar dark(30.0, 30.0, 30.0);

    constexpr double fillOpacity = 0.2;

    cv::Mat3b image;
    cv::cvtColor(snapshot.frame, image, cv::COLOR_GRAY2BGR);

    // Pixel centers scale with the frame, their corners do not
    const auto scaled = [&](const cv::Point2d& p) {
        return cv::Point2d((p.x + 0.5) * snapshot.scale - 0.5, (p.y + 0.5) * snapshot.scale - 0.5);
    };

    std::vector<cv::RotatedRect> ellipses;

    for (const DrawnMatch& d : snapshot.matches) {
        const cv::RotatedRect e(
          scaled(d.ellipse.center), d.ellipse.size * static_cast<float>(snapshot.scale), d.ellipse.angle);

        // Ellipses much larger than the image say nothing, and are expensive to draw
        if (std::isfinite(e.size.width) && e.size.width < 20.0F * static_cast<float>(image.cols)) {
            ellipses.push_back(e);
        }
    }

    // The regions, translucent, so that the image shows through them
    cv::Mat3b filled = image.clone();

    for (const cv::RotatedRect& e : ellipses) {
        cv::ellipse(filled, e, cyan, cv::FILLED, cv::LINE_AA);
    }

    cv::addWeighted(filled, fillOpacity, image, 1.0 - fillOpacity, 0.0, image);

    for (const cv::RotatedRect& e : ellipses) {
        cv::ellipse(image, e, cyan, 1, cv::LINE_AA);
    }

    // A dot with a dark rim, visible on any background
    const auto dot = [&](const cv::Point2d& p, const cv::Scalar& color) {
        cv::circle(image, p, 4, dark, cv::FILLED, cv::LINE_AA);
        cv::circle(image, p, 3, color, cv::FILLED, cv::LINE_AA);
    };

    for (const DrawnMatch& d : snapshot.matches) {
        // The innovation, from the prediction to the match
        cv::line(image, scaled(d.prediction), scaled(d.match), cv::Scalar::all(255.0), 1, cv::LINE_AA);

        dot(scaled(d.seed), magenta);
        dot(scaled(d.prediction), cyan);
        dot(scaled(d.match), amber);
    }

    return image;
}

void
monoslam::Display::post(Snapshot snapshot)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    latest_ = std::move(snapshot);
}

std::optional<monoslam::Snapshot>
monoslam::Display::take()
{
    std::optional<Snapshot> snapshot;

    {
        const std::lock_guard<std::mutex> lock(mutex_);
        std::swap(snapshot, latest_);
    }

    if (snapshot.has_value()) {
        ++numShown_;
    }

    return snapshot;
}
