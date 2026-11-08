#!/usr/bin/env python3
"""Scores a MonoSLAM state stream against the output of analyze_pattern.

The pixel error comes first: the detected corners are checked directly on the images, while a
pose error says nothing about what the camera actually sees. Use with --annotate to look at
the frames as well, a low number is not a proof of good tracking.

The features scored are those MonoSLAM initializes from the pattern, the identifiers 0 to 4, which
are projected with the camera pose and the camera model.

Near the border of the frame the pattern can be fitted with a reprojection error of several pixels,
and its pose is then less reliable than the tracking it is meant to score. With --max-rms, the
frames fitted worse than that, in pixels, are not scored.

Usage: evaluate_tracking.py stream.txt ground_truth.txt camera.json [--max-rms PX]
"""

import json
import re
import sys

import numpy as np


def load_ground_truth(path, max_rms):
    """Returns the camera position and the pixels of the five corners in every frame where the
    pattern was found, and fitted within max_rms pixels."""
    positions, pixels = {}, {}
    frame = None

    with open(path) as file:
        for line in file:
            match = re.match(r"frame_(\d+)\.png OK rms=(\S+) pos=\[(.*?)\]", line)

            if match:
                frame = int(match[1]) if float(match[2]) <= max_rms else None

                if frame is not None:
                    positions[frame] = np.fromstring(match[3], sep=",")

            elif "sel=" in line and frame is not None:
                values = re.search(r"\[(.*)\]", line)[1]
                pixels[frame] = np.fromstring(values, sep=",").reshape(-1, 2)
                frame = None

    return positions, pixels


def load_stream(path):
    """Returns the frames of a stream: index, camera position and orientation (a rotation vector),
    number of features used, and the position of every feature by its identifier."""
    frames = []

    with open(path) as file:
        for line in file:
            fields = line.split()

            if fields[0] == "frame":
                frame = {
                    "index": int(fields[1]),
                    "r": np.array([float(v) for v in fields[2:5]]),
                    "o": np.array([float(v) for v in fields[5:8]]),
                    "used": int(fields[8]),
                    "features": {},
                }

            elif fields[0] == "feature":
                frame["features"][int(fields[1])] = np.array([float(v) for v in fields[3:6]])

            elif fields[0] == "end":
                frames.append(frame)

    return frames


def rotation(o):
    """Rotation matrix of a rotation vector, from camera to world: Rodrigues' formula."""
    angle = np.linalg.norm(o)

    if angle < 1e-12:
        return np.eye(3)

    k = o / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])

    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def project(camera, r, o, point):
    """Pixel of a point, as CameraModel::project, or nan if it is not in view."""
    p = rotation(o).T @ (point - r)

    if p[2] <= 0.0:
        return np.array([np.nan, np.nan])

    x, y = p[0] / p[2], p[1] / p[2]
    r2 = x * x + y * y
    radial = 1.0 + camera["k1"] * r2 + camera["k2"] * r2 * r2 + camera["k3"] * r2 * r2 * r2
    tx = 2.0 * camera["p1"] * x * y + camera["p2"] * (r2 + 2.0 * x * x)
    ty = 2.0 * camera["p2"] * x * y + camera["p1"] * (r2 + 2.0 * y * y)

    u = camera["fx"] * (x * radial + tx) + camera["cx"]
    v = camera["fy"] * (y * radial + ty) + camera["cy"]

    return np.array([u, v])


def main():
    arguments = sys.argv[1:]
    max_rms = np.inf

    if len(arguments) == 5 and arguments[3] == "--max-rms":
        max_rms = float(arguments[4])
        del arguments[3:]

    if len(arguments) != 3:
        sys.exit(__doc__)

    positions, pixels = load_ground_truth(arguments[1], max_rms)
    frames = load_stream(arguments[0])

    with open(arguments[2]) as file:
        camera = json.load(file)

    pixel_errors, pose_errors, losses = [], [], []

    for frame in frames:
        index = frame["index"]

        if index not in pixels:
            continue

        # Features removed from the map are nan, and only the remaining ones are scored
        projected = np.array(
            [
                project(camera, frame["r"], frame["o"], frame["features"][i])
                if i in frame["features"]
                else np.array([np.nan, np.nan])
                for i in range(len(pixels[index]))
            ]
        )

        errors = np.linalg.norm(projected - pixels[index], axis=1)

        if np.all(np.isnan(errors)):
            continue

        pixel_errors.append(np.nanmax(errors))
        pose_errors.append(np.linalg.norm(frame["r"] - positions[index]))

        if pixel_errors[-1] > 5.0:
            losses.append(index)

    pixel_errors = np.array(pixel_errors)
    pose_errors = np.array(pose_errors)

    print(f"Frames: {len(frames)}, scored: {len(pixel_errors)}, coasted: {sum(f['used'] == 0 for f in frames)}")
    print(
        "Worst feature (px): "
        f"median {np.median(pixel_errors):.2f}, p95 {np.percentile(pixel_errors, 95):.2f}, max {pixel_errors.max():.2f}"
    )
    print(
        "Position (cm): "
        f"mean {pose_errors.mean():.2f}, p95 {np.percentile(pose_errors, 95):.2f}, max {pose_errors.max():.2f}"
    )
    print(f"Used features per frame: {np.mean([f['used'] for f in frames]):.2f}")
    print(f"Frames over 5 px: {len(losses)} {losses[:40]}")


if __name__ == "__main__":
    main()
