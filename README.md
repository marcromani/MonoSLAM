# MonoSLAM

A concise and thoroughly documented C++ implementation of monocular EKF SLAM, after the seminal
work of [Davison et al., 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf).
The project is mostly compliant with ISO C++17, and it compiles against OpenCV 5.

It runs at 30 Hz on commodity hardware, check the [sample videos](https://www.youtube.com/watch?v=zg3O0ZN1_rI&list=PLQjxkIPDI8hgJj4gUFwuYeaorkd8JWDpo).

## The code

A small monocular EKF SLAM: one camera, no IMU, initialized from a chessboard and then tracking
the natural features of the scene. The whole system is in `MonoSLAM/`:

| File | What it does |
|---|---|
| `main.cpp` | The loop of every frame, step by step, and the command line |
| `ekf.h/.cpp` | The state layout, the motion model, and the EKF: prediction, observation, the update with 1-point RANSAC, adding and removing features |
| `inverse_depth.h/.cpp` | The inverse depth parametrization of the features and its Jacobians |
| `camera.h/.cpp` | The camera pose and model with lens distortion, its Jacobians, and the rotation helpers |
| `tracking.h/.cpp` | The seeds that follow the image motion of the features, and the search of their warped patches |
| `relocalization.h/.cpp` | Finding the map anywhere in a frame by ORB descriptors, P3P and RANSAC |
| `mapper.h/.cpp` | Adding features where the frame has none, and removing those rarely matched |
| `initialization.h/.cpp` | The first state, from the chessboard |
| `feature.h/.cpp` | The patch of a feature |
| `frame_source.h/.cpp`, `state_stream.h/.cpp` | Input from a camera or a directory, and output of the state |

The EKF state is `[r, e, v, w, y_0, ..., y_N]`: camera position, orientation error, linear and
angular velocity, and stationary features in inverse depth
([Civera, Davison and Montiel, 2008](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_tro2008.pdf)),
`y = (anchor, azimuth, elevation, inverse depth)`, which every feature keeps for its whole
life. The orientation is a rotation matrix kept next to the state, which only holds a small
rotation `e` on top of it, absorbed after every update (an error-state EKF): no quaternion to
normalize. The motion model is a constant velocity model whose velocities decay towards zero, which
fits a hand-held camera better than velocities that persist. The wrong matches are found with
1-point RANSAC
([Civera et al., 2010](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf)), using
only the filter's own models and covariances, and a match is not even proposed if another place in
its search region looks almost as alike.

Every feature is searched in a small region around its seed, which follows the image motion of the
corners around it: the motion model cannot predict a jerk of the hand, but the images can. A
feature that comes back into view after a while is searched around its prediction instead, within
three standard deviations of its innovation, since its seed has drifted in the meantime. New
features are added, as in
[Davison et al., 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf), where
the frame has none, if they are textured enough to be located and do not look like anything around
them, and removed when they stop being matched, unless they have been used long enough to be part
of the map the camera comes back to. After a weak frame the map is looked for where the keyframes
that look most like the frame saw it, small blurred images of frames where tracking was good
([Klein and Murray, 2008](https://doi.org/10.1007/978-3-540-88688-4_59)), and if that fails,
anywhere in the frame by the ORB descriptors every feature keeps of how it has looked
([Williams, Klein and Reid, 2011](https://doi.org/10.1109/TPAMI.2011.41)).

`docs/monoslam.tex` describes the system in detail: the frame loop, the derivation of the
models, every step with its parameters and the evidence behind them, and the current results.

## How to build

Configure the project with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

To enable debug symbols, and AddressSanitizer and UndefinedBehaviorSanitizer for MonoSLAM, use:

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

To build only one target use:

`cmake --build build --target calibration --parallel` (Calibration)

`cmake --build build --target monoslam --parallel` (MonoSLAM)

`cmake --build build --target record_sequence --parallel` (Sequence recorder)

`cmake --build build --target analyze_pattern --parallel` (Ground truth from the chessboard)

Calibrate the camera with a chessboard of 7 x 10 squares of 2.5 cm, then run MonoSLAM with it:

```bash
./build/Calibration/calibration 7 10 2.5 camera
./build/MonoSLAM/monoslam 7 10 2.5 camera
```

The calibration captures a view of the chessboard every two seconds until Escape is pressed, fits
the camera model on the largest frames the camera gives in the same 4:3 shape, 1024 x 768, and saves
it scaled to the 320 x 240 frames that MonoSLAM works on. MonoSLAM refuses a calibration for another
size: the lengths in pixels of its code are chosen for 320 x 240. The calibration accepts these
options after the camera file:

```text
--device D     Camera device (default /dev/video5)
--tolerance T  Largest RMS reprojection error of a view that is kept, in pixels of the camera
               frames (default 3)
```

MonoSLAM accepts these options after the camera file:

```text
--device D    Camera device (default /dev/video5)
--exposure E  Exposure time of the camera, in ms (default 5): fixed, so that the frame rate is too
--images D    Replay the sorted images of directory D instead of a camera
--frames N    Stop after N frames
--stream F    Write the state of every frame to F, for the viewer and evaluate_tracking.py
--annotate D  Save the frames with the features the filter used: their predictions and 99%
              ellipses (cyan), their seeds (magenta), and their matches (amber)
--headless    Do not open any window
--timing      Report every second the frame interval, the decoding and processing times, the
              frames dropped and those shown
```

MonoSLAM asks the camera for 320 x 240 frames, which it gives at about 100 Hz, with a fixed
exposure: with automatic exposure the camera lengthens it in dim light, and lowers the frame rate to
make room for it. Raise `--exposure` if the frames are too dark, up to 7 ms, beyond which the frame
rate falls. The camera keeps the setting after MonoSLAM ends; the calibration turns automatic
exposure back on.

The window shows the frames enlarged to 640 x 480, with the matches drawn on them. It is drawn and
shown by the main thread, and tracking runs in another, which hands it only its last frame: a frame
the window has no time for is skipped, so the window never slows tracking down, whatever its size.
The frames are never copied: each one read is a new image that nothing writes to, which tracking,
the features and the window share. `--annotate` draws every frame in the tracking thread instead.

## Seeing the scene in 3D

`MonoSLAM/tools/viewer.py` shows the scene as the filter sees it in [Rerun](https://rerun.io): the
plane of the pattern, the camera and its trajectory, and the features, red if the last update used
them, yellow if they were in view but not used, gray otherwise, each with the segment of its ray
within two standard deviations of its inverse depth. Every frame stays on the timeline, so any of
them can be looked at again, next to plots of the features used and the position uncertainty.

The viewer needs Rerun and NumPy. Recent Ubuntu releases do not let `pip` install into the system
Python, so install them in a virtual environment, anywhere, and activate it before using the
viewer:

```bash
python3 -m venv ~/rerun-env
source ~/rerun-env/bin/activate
pip install rerun-sdk numpy
```

Feed the viewer while MonoSLAM runs, through a process substitution of bash, or give it a stream
saved to a file. With `--save` it writes a recording instead, to open later with `rerun`:

```bash
./build/MonoSLAM/monoslam 7 10 2.5 camera --stream >(python3 MonoSLAM/tools/viewer.py)
python3 MonoSLAM/tools/viewer.py state.txt
python3 MonoSLAM/tools/viewer.py state.txt --save scene.rrd
```

## Recording a replay sequence

Use the standalone recorder to capture timestamped grayscale frames:

```bash
./build/MonoSLAM/record_sequence camera /dev/video5 /path/to/sequence --frames 300
```

This writes `frame_*.png`, at the 320 x 240 of the calibration, `timestamps.csv` with the capture
time of every frame, and a copy of `camera.json` into the output directory, which must be empty or
not exist yet. With `--frames` or `--seconds` recording starts right away, and otherwise when you
press Space in the preview window. Escape stops it. `--exposure E` sets the exposure time of the
camera, as for MonoSLAM, and `--headless` records without the preview window, with `--frames` or
`--seconds`.

Replay the sequence through the EKF:

```bash
./build/MonoSLAM/monoslam 7 10 2.5 /path/to/sequence/camera \
  --images /path/to/sequence \
  --stream /path/to/state.txt \
  --headless
```

## Evaluating a sequence

`analyze_pattern` detects the whole chessboard in every frame where it can and fits the camera
pose to it, and `MonoSLAM/tools/evaluate_tracking.py` scores a state stream against that, in
pixels first. It needs NumPy, which the environment of the viewer already has:

```bash
./build/MonoSLAM/analyze_pattern /path/to/sequence > ground_truth.txt
python3 MonoSLAM/tools/evaluate_tracking.py /path/to/state.txt ground_truth.txt /path/to/sequence/camera.json
```

The detector misses the blurriest frames, so always check those with `--annotate`. A single run is
not enough to compare two versions: a negligible difference can flip a borderline decision and send
the run elsewhere, so compare several runs, with different initial corners.

## Developer dependencies

Source code formatting depends on:
```
clang-format 23.1.4
cmakefmt 2.3.1
```

Usage:

```bash
clang-format -i Calibration/*.cpp MonoSLAM/*.cpp MonoSLAM/*.h MonoSLAM/tests/*.cpp MonoSLAM/tools/*.cpp
cmakefmt -i .
```

## References

Titles link to a free copy of the paper where there is one, and the DOI to the publisher's version.

1. J. Civera, A. J. Davison and J. M. M. Montiel.
   [Inverse depth parametrization for monocular SLAM](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_tro2008.pdf).
   *IEEE Transactions on Robotics*, 24(5):932–945, 2008.
   DOI: [10.1109/TRO.2008.2003276](https://doi.org/10.1109/TRO.2008.2003276)
2. J. Civera, O. G. Grasa, A. J. Davison and J. M. M. Montiel.
   [1-point RANSAC for extended Kalman filtering: application to real-time structure from motion and visual odometry](https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf).
   *Journal of Field Robotics*, 27(5):609–631, 2010.
   DOI: [10.1002/rob.20345](https://doi.org/10.1002/rob.20345)
3. A. J. Davison, I. D. Reid, N. D. Molton and O. Stasse.
   [MonoSLAM: real-time single camera SLAM](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf).
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 29(6):1052–1067, 2007.
   DOI: [10.1109/TPAMI.2007.1049](https://doi.org/10.1109/TPAMI.2007.1049)
4. E. J. Lefferts, F. L. Markley and M. D. Shuster.
   [Kalman filtering for spacecraft attitude estimation](https://doi.org/10.2514/3.56190).
   *Journal of Guidance, Control, and Dynamics*, 5(5):417–429, 1982.
   DOI: [10.2514/3.56190](https://doi.org/10.2514/3.56190)
5. B. D. Lucas and T. Kanade.
   [An iterative image registration technique with an application to stereo vision](https://ijcai.org/Proceedings/81-2/Papers/017.pdf).
   *Proceedings of the 7th International Joint Conference on Artificial Intelligence*, 674–679, 1981.
6. E. Rublee, V. Rabaud, K. Konolige and G. Bradski.
   [ORB: an efficient alternative to SIFT or SURF](https://web.archive.org/web/20150415004214/http://www.willowgarage.com/sites/default/files/orb_final.pdf).
   *Proceedings of the IEEE International Conference on Computer Vision*, 2564–2571, 2011.
   DOI: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544)
7. J. Shi and C. Tomasi.
   [Good features to track](https://users.cs.duke.edu/~tomasi/papers/shi/shiCvpr94.pdf).
   *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 593–600, 1994.
   DOI: [10.1109/CVPR.1994.323794](https://doi.org/10.1109/CVPR.1994.323794)
8. B. Williams, G. Klein and I. Reid.
   [Automatic relocalization and loop closing for real-time monocular SLAM](https://doi.org/10.1109/TPAMI.2011.41).
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(9):1699–1712, 2011.
   DOI: [10.1109/TPAMI.2011.41](https://doi.org/10.1109/TPAMI.2011.41)
