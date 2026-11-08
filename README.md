# MonoSLAM

A small, well documented C++ implementation of monocular EKF SLAM, based on the work of
[Davison et al., 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf).
The code is mostly ISO C++17 and builds against OpenCV 5.

It runs at 100 Hz on an ordinary laptop.

## The code

One camera, no IMU. The system starts from a chessboard and then tracks the natural features of
the scene. Everything lives in `MonoSLAM/`:

| File | What it does |
|---|---|
| `main.cpp` | The per-frame loop, step by step, and the command line |
| `ekf.h/.cpp` | State layout, motion and measurement models, and the EKF itself: prediction, observation, the update with 1-point RANSAC, adding and removing features |
| `inverse_depth.h/.cpp` | Inverse depth parametrization of the features and its Jacobians |
| `camera.h/.cpp` | Camera pose, camera model with lens distortion, their Jacobians and some rotation helpers |
| `tracking.h/.cpp` | Seeds that follow the image motion of the features, and the patch search |
| `relocalization.h/.cpp` | Finding the map again after tracking fails, from keyframes or ORB descriptors |
| `mapper.h/.cpp` | Adding features where the frame has none, and removing the ones that are rarely matched |
| `initialization.h/.cpp` | The first state, from the chessboard |
| `feature.h/.cpp` | A feature's patch, descriptors and record of matches |
| `frame_source.h/.cpp`, `state_stream.h/.cpp` | Reading from a camera or a directory, and writing the state out |

The EKF state is

$$
x = (r, e, v, \omega, y_1, \dots, y_N),
$$

the camera position $r$, its orientation error $e$, its linear and angular velocities $v$ and
$\omega$, and the features. Features are static points in inverse depth
([Civera, Davison and Montiel, 2008](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_tro2008.pdf)),
$y = (a, \theta, \phi, \rho)$: the anchor $a$, where the camera was when it first saw the point, the
azimuth $\theta$ and elevation $\phi$ of the ray from there, and the inverse depth $\rho$ along it.
They keep this parametrization for their whole life. The orientation is a rotation matrix $R$
kept outside the state, and the state only holds a small rotation $e$ on top of it, so the
orientation is $R \exp([e]_\times)$. After every update $R$ absorbs $e$ (an error-state EKF), and
there is no quaternion to normalize. The motion model is constant velocity, but the velocities
decay towards zero, which suits a hand-held camera better.

Wrong matches are rejected with 1-point RANSAC
([Civera et al., 2010](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf)),
using nothing but the filter's own models and covariances. Before that, a match is dropped if
something else in its search region looks almost the same. Each match is refined to subpixel
precision with Lucas-Kanade
([Baker and Matthews, 2004](https://doi.org/10.1023/B:VISI.0000011205.11775.fd)), and when the
alignment is poor in some direction, along an edge for example, the match gets a larger noise in
that direction.

Each feature is searched in a small region around its seed. The seed follows the image motion of
the corners around it: the motion model can't predict a sudden jerk of the hand, but the images
can. A feature that comes back into view after a while has a stale seed, so it is searched around
its prediction instead, within three standard deviations of its innovation.

New features are added where the frame has none, as in
[Davison et al., 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf).
They have to be textured enough to be located, and must not look like anything around them.
Features that stop being matched are removed, unless they have been used long enough to be part of
the map the camera comes back to.

After a weak frame, the map is looked for with keyframes, small blurred copies of frames where
tracking went well
([Klein and Murray, 2008](https://doi.org/10.1007/978-3-540-88688-4_59)): the features are searched
where the most similar keyframes saw them. If that fails, the whole frame is searched with the ORB
descriptors that every feature collects as its appearance changes
([Williams, Klein and Reid, 2011](https://doi.org/10.1109/TPAMI.2011.41)).

`docs/monoslam.tex` has all the details: the frame loop, the derivation of the models, every step
with its parameters and the evidence behind them, and the current results.

## How far a filter goes

This is about as far as monocular EKF SLAM goes. When the camera turns away from the mapped part
of the scene, every feature in view is new, with a very uncertain depth, and the filter linearizes
its updates at depths that later turn out wrong. It can't go back on those updates, so it becomes
confident about a pose that is off. In a simulation where all the models are exact, the pose error
ends up over a hundred times larger than the filter thinks, while features that enter at their
true depth keep it consistent. None of the classic remedies helps: second-order terms, Schmidt
updates, first-estimate Jacobians, an iterated EKF, delayed initialization. A bundle adjustment
over the same matches does recover the motion, so the information is in the images.

That is what [Strasdat, Montiel and Davison, 2012](https://doi.org/10.1016/j.imavis.2012.02.009)
found, and what ended the filtering era of visual SLAM. The next step would be tracking against a
map kept by bundle adjustment, as PTAM and its successors do, not a better filter. The last section
of `docs/monoslam.tex` has the evidence.

## How to build

Configure the project with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

For debug symbols, plus AddressSanitizer and UndefinedBehaviorSanitizer on MonoSLAM, use:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Build with:

```bash
cmake --build build --parallel
```

Run the Jacobian and filter tests with:

```bash
ctest --test-dir build --output-on-failure
```

You can also build a single target: the calibration, MonoSLAM, the sequence recorder or the
chessboard ground truth:

```bash
cmake --build build --target calibration --parallel
cmake --build build --target monoslam --parallel
cmake --build build --target record_sequence --parallel
cmake --build build --target analyze_pattern --parallel
```

Calibrate the camera with a chessboard of 7 x 10 squares of 2.5 cm, then run MonoSLAM:

```bash
./build/Calibration/calibration 7 10 2.5 camera --device /dev/video4
./build/MonoSLAM/monoslam 7 10 2.5 camera --device /dev/video4
```

All three programs need the camera device, given with `--device`. `v4l2-ctl --list-devices` lists
the cameras; for each camera, the first device is the one that gives frames.

The calibration grabs a view of the chessboard every two seconds until you press Escape. It fits
the camera model at 640 x 480, the largest frames the camera gives with the same field of view, and
saves it scaled down to the 320 x 240 that MonoSLAM uses. Note that a larger mode with the same
aspect ratio doesn't necessarily have the same field of view: at 1024 x 768 this camera sees 12%
more in each direction. MonoSLAM refuses a calibration made for any other size, since the pixel
lengths in the code are tuned for 320 x 240. The calibration takes these options after the camera
file:

```text
--device D     Camera device, such as /dev/video4 (required)
--tolerance T  Largest RMS reprojection error of a view that is kept, in
               pixels of the camera frames (default 1.5)
```

MonoSLAM takes these options after the camera file:

```text
--device D    Camera device, such as /dev/video4 (required, unless --images)
--exposure E  Exposure time of the camera, in ms (default 5): fixed, so that
              the frame rate is too
--images D    Replay the sorted images of directory D instead of a camera
--frames N    Stop after N frames
--stream F    Write the state of every frame to F, for the viewer and
              evaluate_tracking.py
--annotate D  Save the frames with the features the filter used: their
              predictions and 99% ellipses (cyan), their seeds (magenta), and
              their matches (amber)
--headless    Do not open any window
--timing      Report every second the frame interval, the decoding and
              processing times, the frames dropped and those shown
```

MonoSLAM asks the camera for 320 x 240 frames, which it delivers at about 100 Hz, with a fixed
exposure. With automatic exposure the camera lengthens the exposure in dim light and drops the
frame rate to fit it. If the frames are too dark, raise `--exposure`, up to 7 ms; beyond that the
frame rate falls. The camera remembers the setting after MonoSLAM exits, and the calibration turns
automatic exposure back on.

The window shows the frames enlarged to 640 x 480, with the matches drawn on top. The main thread
draws the window and tracking runs in a separate thread, which only hands over its latest frame.
If the window can't keep up, it skips frames, so it never slows tracking down, whatever its size.
Frames are never copied either: every frame read is a fresh image that nobody writes to, shared by
tracking, the features and the window. With `--annotate`, the tracking thread draws every frame
itself.

## Seeing the scene in 3D

`MonoSLAM/tools/viewer.py` shows the scene as the filter sees it, in [Rerun](https://rerun.io).
You get the chessboard plane, the camera and its trajectory, and the features: red if the last
update used them, yellow if they were in view but not used, and gray otherwise. Each feature is
drawn as the segment of its ray within two standard deviations of its inverse depth. All the frames
stay on the timeline, so you can go back to any of them, and there are plots of the number of
features used and of the position uncertainty.

The viewer needs Rerun and NumPy. Recent Ubuntu releases don't let `pip` install into the system
Python, so install them in a virtual environment, anywhere you like, and activate it before using
the viewer:

```bash
python3 -m venv ~/rerun-env
source ~/rerun-env/bin/activate
pip install rerun-sdk numpy
```

You can feed the viewer live, through bash process substitution, or give it a saved stream. With
`--save` it writes a recording instead, which you can open later with `rerun`:

```bash
./build/MonoSLAM/monoslam 7 10 2.5 camera --device /dev/video4 \
  --stream >(python3 MonoSLAM/tools/viewer.py)
python3 MonoSLAM/tools/viewer.py state.txt
python3 MonoSLAM/tools/viewer.py state.txt --save scene.rrd
```

## Recording a sequence

Use the recorder to capture timestamped grayscale frames:

```bash
./build/MonoSLAM/record_sequence camera /path/to/sequence \
  --device /dev/video4 \
  --frames 300
```

It writes the frames (`frame_*.png`, at 320 x 240), `timestamps.csv` with the capture time of each
frame, and a copy of `camera.json`. The output directory must be empty or not exist yet. The
recorder takes these options after the directory:

```text
--device D    Camera device, such as /dev/video4 (required)
--exposure E  Exposure time of the camera, in ms (default 5), as for MonoSLAM
--frames N    Record N frames, starting right away
--seconds S   Record S seconds, starting right away
--headless    Do not open any window, only with --frames or --seconds
```

Without `--frames` or `--seconds`, recording starts when you press Space in the preview window,
and Escape stops it.

To replay the sequence through the EKF:

```bash
./build/MonoSLAM/monoslam 7 10 2.5 /path/to/sequence/camera \
  --images /path/to/sequence \
  --stream /path/to/state.txt \
  --headless
```

## Evaluating a sequence

`analyze_pattern` finds the whole chessboard in every frame where it can and computes the camera
pose from it. `MonoSLAM/tools/evaluate_tracking.py` then scores a state stream against those poses,
mainly in pixels. It needs NumPy, which the viewer environment already has:

```bash
./build/MonoSLAM/analyze_pattern /path/to/sequence > ground_truth.txt
python3 MonoSLAM/tools/evaluate_tracking.py /path/to/state.txt \
  ground_truth.txt \
  /path/to/sequence/camera.json
```

Near the border of the frame the chessboard can be fitted with a reprojection error of several
pixels, and its pose is then worse than the tracking it scores. `--max-rms 1.5` after the camera
file leaves those frames out. The detector also misses the blurriest frames, so check those with
`--annotate`.

Don't compare two versions on a single run. A tiny difference can flip a borderline decision and
send the whole run somewhere else, so compare several runs, each starting from different chessboard
corners.

## Developer dependencies

Source code formatting depends on:

```text
clang-format 23.1.4
cmakefmt 2.3.1
```

Usage:

```bash
clang-format -i Calibration/*.cpp \
  MonoSLAM/*.cpp MonoSLAM/*.h \
  MonoSLAM/tests/*.cpp \
  MonoSLAM/tools/*.cpp
cmakefmt -i .
```

## References

Titles link to a free copy of the paper when there is one, and the DOI to the publisher's version.

1. S. Baker and I. Matthews.
   [Lucas-Kanade 20 years on: a unifying framework](https://doi.org/10.1023/B:VISI.0000011205.11775.fd).
   *International Journal of Computer Vision*, 56(3):221–255, 2004.
   DOI: [10.1023/B:VISI.0000011205.11775.fd](https://doi.org/10.1023/B:VISI.0000011205.11775.fd)
2. J. Civera, A. J. Davison and J. M. M. Montiel.
   [Inverse depth parametrization for monocular SLAM](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_tro2008.pdf).
   *IEEE Transactions on Robotics*, 24(5):932–945, 2008.
   DOI: [10.1109/TRO.2008.2003276](https://doi.org/10.1109/TRO.2008.2003276)
3. J. Civera, O. G. Grasa, A. J. Davison and J. M. M. Montiel.
   [1-point RANSAC for extended Kalman filtering: application to real-time structure from motion and visual odometry](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf).
   *Journal of Field Robotics*, 27(5):609–631, 2010.
   DOI: [10.1002/rob.20345](https://doi.org/10.1002/rob.20345)
4. A. J. Davison, I. D. Reid, N. D. Molton and O. Stasse.
   [MonoSLAM: real-time single camera SLAM](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf).
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 29(6):1052–1067, 2007.
   DOI: [10.1109/TPAMI.2007.1049](https://doi.org/10.1109/TPAMI.2007.1049)
5. E. J. Lefferts, F. L. Markley and M. D. Shuster.
   [Kalman filtering for spacecraft attitude estimation](https://doi.org/10.2514/3.56190).
   *Journal of Guidance, Control, and Dynamics*, 5(5):417–429, 1982.
   DOI: [10.2514/3.56190](https://doi.org/10.2514/3.56190)
6. B. D. Lucas and T. Kanade.
   [An iterative image registration technique with an application to stereo vision](https://ijcai.org/Proceedings/81-2/Papers/017.pdf).
   *Proceedings of the 7th International Joint Conference on Artificial Intelligence*, 674–679, 1981.
7. E. Rublee, V. Rabaud, K. Konolige and G. Bradski.
   [ORB: an efficient alternative to SIFT or SURF](https://web.archive.org/web/20150415004214/http://www.willowgarage.com/sites/default/files/orb_final.pdf).
   *Proceedings of the IEEE International Conference on Computer Vision*, 2564–2571, 2011.
   DOI: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544)
8. J. Shi and C. Tomasi.
   [Good features to track](https://users.cs.duke.edu/~tomasi/papers/shi/shiCvpr94.pdf).
   *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 593–600, 1994.
   DOI: [10.1109/CVPR.1994.323794](https://doi.org/10.1109/CVPR.1994.323794)
9. H. Strasdat, J. M. M. Montiel and A. J. Davison.
   [Visual SLAM: why filter?](https://doi.org/10.1016/j.imavis.2012.02.009).
   *Image and Vision Computing*, 30(2):65–77, 2012.
   DOI: [10.1016/j.imavis.2012.02.009](https://doi.org/10.1016/j.imavis.2012.02.009)
10. B. Williams, G. Klein and I. Reid.
    [Automatic relocalization and loop closing for real-time monocular SLAM](https://doi.org/10.1109/TPAMI.2011.41).
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(9):1699–1712, 2011.
    DOI: [10.1109/TPAMI.2011.41](https://doi.org/10.1109/TPAMI.2011.41)
