#ifndef MONOSLAM_MAPPER_H
#define MONOSLAM_MAPPER_H

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "ekf.h"
#include "feature.h"
#include "tracking.h"

namespace monoslam {

/*
 * Decides which features the map has: it adds new ones where too few are in view, and removes those that are rarely
 * matched. The state of the filter, the features and the seeds of the flow tracker are kept in the same order.
 */
class Mapper
{
  public:
    /*
     * firstId    Identifier of the first feature the mapper adds
     */
    explicit Mapper(std::size_t firstId);

    /*
     * Counts, for every feature that was searched, whether the filter used its match. A feature that comes back
     * into view after a while starts its count anew, and is returning until the filter uses it again.
     *
     * features    All the features
     * searched    Features that were searched in this frame
     * used        Features whose matches the filter used
     */
    void recordMatches(std::vector<Feature>& features,
                       const std::vector<std::size_t>& searched,
                       const std::vector<std::size_t>& used) const;

    /*
     * Keeps the descriptors of the features used in this frame, if they look different enough from how they have looked
     * so far, so that they can be found from other points of view.
     *
     * frame        Current frame
     * features     All the features
     * used         Features whose matches the filter used
     * positions    Image position of all the features
     */
    void recordAppearance(const cv::Mat1b& frame,
                          std::vector<Feature>& features,
                          const std::vector<std::size_t>& used,
                          const std::vector<cv::Point2f>& positions) const;

    /*
     * Removes the features used in less than half of their recent attempts, once they have had a few. This is the rule
     * of Davison et al., but over the last attempts only, so that a bad feature is removed within a third of a second.
     * Such features are usually not salient enough, occluded, on the edge of an object, or on something that moves.
     * The corners of the pattern, and the features used in at least 30 frames, are never removed: the camera relies
     * on them when it comes back.
     *
     * ekf         Filter
     * features    All the features
     * tracker     Flow tracker
     */
    void removeUnreliable(EKF& ekf, std::vector<Feature>& features, FlowTracker& tracker) const;

    /*
     * Adds new features where the frame has none, while fewer than a dozen features in view were used in this frame,
     * to keep their density about constant over it, as Davison et al. do. Features in view that were not found do
     * not count, so that they cannot keep new ones out. A box a quarter of the size of the frame is placed where it has
     * no feature inside. The most salient corner in it is added, away from the border of the frame and
     * from the features near the box. Corners must also be textured enough to be located, which a blank wall is not.
     * They must also be distinctive: a corner that looks like something within its search region would be confused
     * with it, as happens on any repeated texture. Must be called after recordMatches, which records the features used
     * in this frame.
     *
     * frame       Current frame
     * camera      Camera model
     * ekf         Filter
     * features    All the features
     * tracker     Flow tracker
     */
    void addFeatures(const cv::Mat1b& frame,
                     const CameraModel& camera,
                     EKF& ekf,
                     std::vector<Feature>& features,
                     FlowTracker& tracker);

  private:
    std::size_t nextId_;  // Identifier of the next feature to add
};

}

#endif
