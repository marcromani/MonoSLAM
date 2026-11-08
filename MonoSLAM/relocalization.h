#ifndef MONOSLAM_RELOCALIZATION_H
#define MONOSLAM_RELOCALIZATION_H

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "ekf.h"
#include "feature.h"

/*
 * Finding the map again when tracking fails, without the state of the camera, which may be wrong.
 * The features are first searched where the keyframes that look most like the frame saw them
 * (Klein and Murray, 2008), and if that fails, anywhere in the frame by their appearance alone
 * (Williams, Klein and Reid, 2011). Either way, a pose is fitted to the features found, and only
 * accepted if the patches of the features are found where it projects them.
 *
 * The appearance of a feature is an ORB descriptor (Rublee et al., 2011): 256 binary comparisons of
 * pixel pairs of the patch around it, rotated with the patch, so that it does not change when the
 * image rotates, and little with moderate changes of scale and viewpoint.
 */
namespace monoslam {

/*
 * Returns the descriptors of some pixels of a frame, one per row, and removes the pixels too close
 * to the border to have one.
 *
 * frame        Grayscale frame
 * positions    Pixels
 */
cv::Mat1b
describe(const cv::Mat1b& frame, std::vector<cv::Point2f>& positions);

/*
 * Returns the descriptor of a pixel, or an empty matrix if it is too close to the border.
 */
cv::Mat1b
describe(const cv::Mat1b& frame, const cv::Point2d& position);

/*
 * Features of the map found in a frame, and the pose that explains them.
 */
struct Relocalization
{
    cv::Vec3d r;    // Camera position
    cv::Matx33d R;  // Camera orientation, from camera to world

    std::vector<std::size_t> features;  // Features found
    std::vector<cv::Point2d> pixels;    // Where they were found
};

/*
 * Matches the corners of a frame to the well located features by their descriptors, and fits a
 * pose to the matches with P3P and RANSAC, taking the positions of the features as exact. The pose
 * is accepted if the patches of the features are also found where it projects them. Returns
 * nothing if too few features agree on a pose.
 *
 * frame       Current frame
 * features    All the features
 * camera      Camera model
 * ekf         Filter
 */
std::optional<Relocalization>
matchGlobally(const cv::Mat1b& frame, const std::vector<Feature>& features, const CameraModel& camera, const EKF& ekf);

/*
 * Frames where tracking was good, small and blurred, with their pose, as in the relocalization of
 * Klein and Murray (2008). A frame that looks like one of them was taken from near its pose, which
 * tells where to look for the features. Unlike the descriptors, it needs no corner where the
 * feature is, and the patches it searches with are those that track the features.
 */
class Keyframes
{
  public:
    /*
     * Keeps a frame if it was taken far from every keyframe, and enough features were used in it.
     *
     * frame      Current frame
     * pose       Camera pose, after the update
     * numUsed    Features used in the frame
     */
    void add(const cv::Mat1b& frame, const CameraPose& pose, std::size_t numUsed);

    /*
     * Searches the well located features, other than those of the pattern, where each of the
     * keyframes that look most like a frame would see them, shifted by the turn of the camera, and
     * fits a pose to them as matchGlobally does. Returns nothing if too few features agree on a
     * pose.
     *
     * frame       Current frame
     * features    All the features
     * camera      Camera model
     * ekf         Filter
     */
    std::optional<Relocalization> match(const cv::Mat1b& frame,
                                        const std::vector<Feature>& features,
                                        const CameraModel& camera,
                                        const EKF& ekf) const;

  private:
    struct Keyframe
    {
        cv::Mat1f thumbnail;  // Shrunk and blurred, with zero mean
        CameraPose pose;
    };

    std::vector<Keyframe> keyframes_;
};

}

#endif
