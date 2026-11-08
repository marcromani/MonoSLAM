#!/usr/bin/env python3
"""Shows the scene as the filter sees it, in Rerun, from the state that monoslam writes with
--stream (see state_stream.h). It reads the state from a file given as its argument, or from its
standard input, so that it can be fed while monoslam runs, through a process substitution:

    monoslam 7 10 2.5 camera --device /dev/video4 --stream >(python3 MonoSLAM/tools/viewer.py)

The world is the frame of the calibration pattern, which lies on the plane z = 0. Shown are the
camera and its trajectory, and the features: red if the last update used them, yellow if they were
in view but not used, gray otherwise, with the segment of their ray where they are within two
standard deviations of their inverse depth. Every frame is kept on the timeline, so that any of
them can be looked at again.

With --save FILE, it writes the recording to FILE instead of opening the viewer, to be opened later
with rerun FILE.

Usage: viewer.py [stream.txt] [--save FILE]
"""

import sys
from pathlib import Path

import numpy as np
import rerun as rr

COLORS = {0: (128, 128, 128), 1: (255, 200, 0), 2: (255, 40, 40)}  # By status

GRID_EXTENT = 60.0  # Of the grid on the plane of the pattern, in cm
GRID_STEP = 10.0

TRAJECTORY_CHUNK = 200  # Points of the trajectory logged together: logging all of them every frame grows quadratically


def frames(stream):
    """Yields the frames of a stream: index, camera position and orientation (a rotation vector),
    number of features used, position standard deviation, and the features."""
    frame = None

    for line in stream:
        fields = line.split()

        if not fields:
            continue

        if fields[0] == "frame":
            values = [float(v) for v in fields[2:]]
            frame = dict(index=int(fields[1]), r=values[0:3], o=np.array(values[3:6]), used=values[6], std=values[7],
                         features=[])

        elif fields[0] == "feature" and frame is not None:
            values = [float(v) for v in fields[3:]]
            frame["features"].append((int(fields[2]), values[0:3], values[3:6], values[6:9]))

        elif fields[0] == "end" and frame is not None:
            yield frame
            frame = None


def log_world():
    """Logs what does not change: the axes of the world and the grid on the plane of the pattern."""
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/axes", rr.Arrows3D(vectors=10.0 * np.eye(3), colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]),
           static=True)

    ticks = np.arange(-GRID_EXTENT, GRID_EXTENT + GRID_STEP, GRID_STEP)
    lines = [[(t, -GRID_EXTENT, 0.0), (t, GRID_EXTENT, 0.0)] for t in ticks]
    lines += [[(-GRID_EXTENT, t, 0.0), (GRID_EXTENT, t, 0.0)] for t in ticks]
    rr.log("world/grid", rr.LineStrips3D(lines, colors=(70, 70, 70), radii=0.05), static=True)

    # The frustum of the camera, about that of the calibration
    rr.log("world/camera", rr.Pinhole(width=320, height=240, focal_length=333.0, camera_xyz=rr.ViewCoordinates.RDF,
                                      image_plane_distance=5.0), static=True)


def log_frame(frame, trajectory):
    """Logs the camera, its trajectory and the features of a frame, at its index on the timeline. The
    trajectory is logged in chunks, each its own entity, and every frame only logs the last one, so
    that the data sent per frame does not grow with the length of the run."""
    rr.set_time("frame", sequence=frame["index"])

    # The camera, from camera to world, with the axes of an image: x right, y down, z forward
    angle = np.linalg.norm(frame["o"])
    axis = frame["o"] / angle if angle > 0.0 else np.array([0.0, 0.0, 1.0])

    rr.log("world/camera", rr.Transform3D(translation=frame["r"], rotation=rr.RotationAxisAngle(axis, radians=angle)))

    # A new chunk starts where the last one ends, so that the trajectory has no gaps
    if len(trajectory["points"]) == TRAJECTORY_CHUNK:
        trajectory["points"] = trajectory["points"][-1:]
        trajectory["chunk"] += 1

    trajectory["points"].append(frame["r"])
    rr.log(f"world/trajectory/{trajectory['chunk']}",
           rr.LineStrips3D([trajectory["points"]], colors=(0, 200, 255), radii=0.15))

    if frame["features"]:
        status, points, near, far = zip(*frame["features"])
        colors = [COLORS[s] for s in status]

        rr.log("world/features", rr.Points3D(points, colors=colors, radii=0.6))
        rr.log("world/rays", rr.LineStrips3D(list(zip(near, far)), colors=colors, radii=0.1))

    rr.log("plots/used", rr.Scalars(frame["used"]))
    rr.log("plots/position_std", rr.Scalars(frame["std"]))


def main():
    arguments = sys.argv[1:]
    save = None

    if "--save" in arguments:
        i = arguments.index("--save")

        if i + 1 >= len(arguments):
            sys.exit(__doc__)

        save = arguments[i + 1]
        del arguments[i : i + 2]

    if len(arguments) > 1:
        sys.exit(__doc__)

    rr.init("monoslam")

    # In a virtual environment the viewer is installed next to its Python, which may not be in PATH
    if save is None:
        viewer = Path(sys.executable).parent / "rerun"
        rr.spawn(executable_path=str(viewer) if viewer.exists() else None)
    else:
        rr.save(save)

    log_world()

    stream = open(arguments[0]) if arguments else sys.stdin
    trajectory = dict(points=[], chunk=0)

    for frame in frames(stream):
        log_frame(frame, trajectory)


if __name__ == "__main__":
    main()
