#!/usr/bin/env python3
"""Shows the scene as the filter sees it, in Rerun, from the state that monoslam writes with --stream (see
state_stream.h). It reads the state from a file given as its argument, or from its standard input, so that it can be fed
while monoslam runs, through a process substitution:

    monoslam 7 10 2.5 camera --device /dev/video4 --stream >(python3 MonoSLAM/tools/viewer.py)

The world is the frame of the calibration pattern, which lies on the plane z = 0. Shown are the camera
and its trajectory, and the features: red if the last update used them, yellow if they were in view but not used, gray
otherwise, with the segment of their ray where they are within two standard deviations of their inverse depth,
and their patch, on the plane that the tracker uses to warp it. Next to the scene is the frame, undistorted,
with the matches the filter used drawn as in the monoslam window; hovering a pixel of it shows its ray in the scene.
Every frame is kept on the timeline, at the time it was captured, so that the recording plays at the speed of the camera
and any frame can be looked at again.

While monoslam runs, Rerun shows each frame as soon as it arrives, so the viewer logs the frames it reads
from its standard input no faster than they were captured: monoslam replays a recorded sequence as fast as it can,
and the scene would play faster than the camera moved. monoslam then waits for the viewer, but no frame is dropped
or changed. A stream given as a file is read at once, and plays at the speed of the camera when replayed in Rerun.

With --save FILE, it writes the recording to FILE instead of opening the viewer, to be opened later with rerun FILE.

Usage: viewer.py [stream.txt] [--save FILE]
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

COLORS = {0: (128, 128, 128), 1: (255, 200, 0), 2: (255, 40, 40)}  # By status

# The matches, as the monoslam window draws them
CYAN, MAGENTA, AMBER, WHITE = (60, 200, 230), (230, 60, 200), (255, 170, 0), (255, 255, 255)
ELLIPSE_POINTS = 48

ATLAS_COLUMNS = 32  # Patches per row of the texture that holds them all

TRAJECTORY_CHUNK = 200  # Points of the trajectory logged together: logging all of them every frame grows quadratically


def records(stream):
    """Yields the records of a stream, as (kind, data): the camera, as its size and intrinsic matrix, the patch
    of every new feature, as its id and pixels, and every frame, as its index, camera position and orientation (a
    rotation vector), number of features used, position standard deviation, capture time, image, features,
    and matches."""
    frame = None

    for line in stream:
        fields = line.split()

        if not fields:
            continue

        if fields[0] == "camera":
            width, height = int(fields[1]), int(fields[2])
            fx, fy, cx, cy = (float(v) for v in fields[3:7])
            yield "camera", (width, height, np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))

        elif fields[0] == "patch":
            size = int(fields[2])
            yield "patch", (int(fields[1]), np.frombuffer(base64.b64decode(fields[3]), np.uint8).reshape(size, size))

        elif fields[0] == "frame":
            values = [float(v) for v in fields[2:]]
            frame = {
                "index": int(fields[1]),
                "r": values[0:3],
                "o": np.array(values[3:6]),
                "used": values[6],
                "std": values[7],
                "time": values[8],
                "image": None,
                "features": [],
                "matches": [],
            }

        elif fields[0] == "image" and frame is not None:
            frame["image"] = base64.b64decode(fields[1])

        elif fields[0] == "feature" and frame is not None:
            values = [float(v) for v in fields[3:]]
            frame["features"].append(
                (
                    int(fields[1]),
                    int(fields[2]),
                    values[0:3],
                    values[3:6],
                    values[6:9],
                    np.reshape(values[9:21], (4, 3)),
                )
            )

        elif fields[0] == "match" and frame is not None:
            values = [float(v) for v in fields[1:]]
            e = values[6:9]
            frame["matches"].append((values[0:2], values[2:4], values[4:6], np.array([[e[0], e[1]], [e[1], e[2]]])))

        elif fields[0] == "end" and frame is not None:
            yield "frame", frame
            frame = None


def send_layout(fps):
    """Sends the layout of the viewer: the scene, and next to it the frame and the plots, with the time panel
    on the time of the frames, and fps the rate at which the frame timeline plays. Only once frames have been logged,
    since before that the viewer does not know the timeline, and warns about it."""
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                # Orbits about the origin of the pattern, not the center of the scene, which moves as it grows
                rrb.Spatial3DView(
                    origin="world",
                    name="Scene",
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=(-40.0, -60.0, 50.0),
                        look_target=(0.0, 0.0, 0.0),
                        eye_up=(0.0, 0.0, 1.0),
                    ),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="world/camera", name="Frame"),
                    rrb.TimeSeriesView(origin="plots", name="Filter"),
                ),
                column_shares=[2, 1],
            ),
            rrb.TimePanel(timeline="time", fps=fps),
        )
    )


def log_world(width, height, K):
    """Logs what does not change: the axes of the world, and the camera, with the intrinsic matrix of its undistorted
    frames."""
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/axes", rr.Arrows3D(vectors=10.0 * np.eye(3), colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]), static=True
    )

    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=K, width=width, height=height, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=5.0
        ),
        static=True,
    )


def log_image(image, matches):
    """Logs the frame with the matches the filter used over it, as the monoslam window draws them: the predicted
    measurement (cyan) with its 99% ellipse, the seed (magenta), the match (amber), and the innovation,
    from the prediction to the match (white). All of it is one row of one entity, so that the viewer never shows
    the matches of a frame over another frame, as it could with separate entities, which reach it at different times.
    The frame is decoded here, so that the viewer has nothing to decode before it can show it."""
    angles = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_POINTS)
    circle = np.stack([np.cos(angles), np.sin(angles)])

    strips, strip_colors = [], []

    for prediction, _, match, E in matches:
        # Ellipses much larger than the frame carry no information, as in the monoslam window
        if np.all(np.isfinite(E)) and np.trace(E) < 1e6:
            strips.append(np.array(prediction) + (np.linalg.cholesky(E) @ circle).T)
            strip_colors.append(CYAN)

        strips.append([prediction, match])
        strip_colors.append(WHITE)

    # Drawn in this order, the matches on top
    predictions, seeds, found, _ = zip(*matches) if matches else ([], [], [], [])
    points = [*seeds, *predictions, *found]
    colors = [MAGENTA] * len(seeds) + [CYAN] * len(predictions) + [AMBER] * len(found)

    pixels = np.asarray(Image.open(io.BytesIO(image)))

    rr.log(
        "world/camera/image",
        rr.Image(pixels, color_model="L"),
        rr.LineStrips2D(strips, colors=strip_colors, radii=0.5),
        rr.Points2D(points, colors=colors, radii=1.5),
    )


class Scene:
    """What the viewer keeps from frame to frame."""

    def __init__(self):
        self.start = None  # Time of the first frame
        self.trajectory = []  # Points of the last chunk of the trajectory
        self.chunk = 0  # Its index
        self.patches = {}  # Of every feature, by id
        self.textured = None  # The features whose patches the texture of the mesh holds, in order

    def log_frame(self, frame):
        """Logs the camera, its frame, its trajectory and the features of a frame, at its index and at its time since
        the first frame. The trajectory is logged in chunks, each its own entity, and every frame only logs the last
        one, so that the data sent per frame does not grow with the length of the run."""
        if self.start is None:
            self.start = frame["time"]

        rr.set_time("frame", sequence=frame["index"])
        rr.set_time("time", duration=frame["time"] - self.start)

        # The camera, from camera to world, with the axes of an image: x right, y down, z forward
        angle = np.linalg.norm(frame["o"])
        axis = frame["o"] / angle if angle > 0.0 else np.array([0.0, 0.0, 1.0])

        rotation = rr.RotationAxisAngle(axis, radians=angle)
        rr.log("world/camera", rr.Transform3D(translation=frame["r"], rotation=rotation))

        if frame["image"] is not None:
            log_image(frame["image"], frame["matches"])

        # A new chunk starts where the last one ends, so that the trajectory has no gaps
        if len(self.trajectory) == TRAJECTORY_CHUNK:
            self.trajectory = self.trajectory[-1:]
            self.chunk += 1

        self.trajectory.append(frame["r"])
        rr.log(f"world/trajectory/{self.chunk}", rr.LineStrips3D([self.trajectory], colors=(0, 200, 255), radii=0.15))

        if frame["features"]:
            _, status, points, near, far, _ = zip(*frame["features"])
            colors = [COLORS[s] for s in status]

            rr.log("world/features", rr.Points3D(points, colors=colors, radii=0.3))
            rr.log("world/rays", rr.LineStrips3D(list(zip(near, far)), colors=colors, radii=0.1))
            self.log_patches(frame["features"])

        rr.log("plots/used", rr.Scalars(frame["used"]))
        rr.log("plots/position_std", rr.Scalars(frame["std"]))

    def log_patches(self, features):
        """Logs the patches of the features as one mesh, a textured quad each, with all the patches in one texture,
        ATLAS_COLUMNS to a row. The texture only changes when the features do, so it is logged only then,
        and the corners every frame."""
        ids = [feature[0] for feature in features]
        corners = np.concatenate([feature[5] for feature in features])

        if ids == self.textured:
            rr.log("world/patches", rr.Mesh3D.from_fields(vertex_positions=corners))
            return

        size = next(iter(self.patches.values())).shape[0]
        rows = (len(ids) + ATLAS_COLUMNS - 1) // ATLAS_COLUMNS
        atlas = np.zeros((rows * size, ATLAS_COLUMNS * size), np.uint8)

        texcoords = []

        for k, id in enumerate(ids):
            row, column = divmod(k, ATLAS_COLUMNS)
            atlas[row * size : (row + 1) * size, column * size : (column + 1) * size] = self.patches[id]

            # Inset by half a texel, so that the neighbors in the texture do not leak onto the edges
            u0, v0 = (column * size + 0.5) / atlas.shape[1], (row * size + 0.5) / atlas.shape[0]
            u1, v1 = ((column + 1) * size - 0.5) / atlas.shape[1], ((row + 1) * size - 0.5) / atlas.shape[0]

            texcoords.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

        quads = 4 * np.arange(len(ids))[:, None]
        triangles = np.concatenate([quads + [0, 1, 2], quads + [0, 2, 3]])

        rr.log(
            "world/patches",
            rr.Mesh3D(
                vertex_positions=corners,
                vertex_texcoords=texcoords,
                triangle_indices=triangles,
                albedo_texture=np.dstack([atlas] * 3),
            ),
        )
        self.textured = ids


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
    rr.set_log_time_enabled(False)  # When the frames are logged is meaningless: their capture time is

    # In a virtual environment the viewer is installed next to its Python, which may not be in PATH
    if save is None:
        viewer = Path(sys.executable).parent / "rerun"
        rr.spawn(executable_path=str(viewer) if viewer.exists() else None)
    else:
        rr.save(save)

    if arguments:
        with open(arguments[0]) as stream:
            show(stream, paced=False)
    else:
        show(sys.stdin, paced=save is None)


def show(stream, paced):
    """Logs the records of a stream. If paced, the frames are logged no faster than they were captured."""
    scene = Scene()
    first = None  # Index and capture time of the first frame, and the time it was logged
    layout = False  # Whether the layout was sent

    for kind, data in records(stream):
        if kind == "camera":
            log_world(*data)

        elif kind == "patch":
            scene.patches[data[0]] = data[1]

        else:
            if first is None:
                first = data["index"], data["time"], time.monotonic()

            if paced:
                time.sleep(max(0.0, first[2] + data["time"] - first[1] - time.monotonic()))

            scene.log_frame(data)

            # The rate of the frames, once a second of them has been logged, or all of them
            elapsed = data["time"] - first[1]

            if not layout and elapsed >= 1.0:
                send_layout(round((data["index"] - first[0]) / elapsed))
                layout = True

    if first is not None and not layout:
        send_layout(round((data["index"] - first[0]) / elapsed) if elapsed > 0.0 else None)


if __name__ == "__main__":
    main()
