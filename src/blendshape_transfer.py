from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class BlendShapeValidationError(ValueError):
    pass


def _vertices(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise BlendShapeValidationError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise BlendShapeValidationError(f"{name} contains NaN or infinity")
    return array


def _faces(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 3:
        raise BlendShapeValidationError(f"{name} must have shape (M, K), K >= 3")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.equal(array, np.floor(array)).all():
            raise BlendShapeValidationError(f"{name} must contain integer indices")
        array = array.astype(np.int64)
    else:
        array = array.astype(np.int64, copy=False)
    if (array < 0).any():
        raise BlendShapeValidationError(f"{name} contains negative indices")
    return array


def topology_sha256(faces: np.ndarray) -> str:
    canonical = np.ascontiguousarray(faces.astype("<i8", copy=False))
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def transfer_same_topology(
    source_base: Any,
    source_shape: Any,
    target_base: Any,
    *,
    source_faces: Any | None = None,
    target_faces: Any | None = None,
    confirmed_same_topology: bool = False,
    max_displacement: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transfer vertex offsets only when topology and vertex order are explicit.

    This is a deterministic baseline, not an ML model and not a
    topology-independent retargeter. Face arrays are the preferred proof. When
    they are unavailable, the caller must explicitly set confirmed_same_topology.
    """

    source_base_array = _vertices("source_base", source_base)
    source_shape_array = _vertices("source_shape", source_shape)
    target_base_array = _vertices("target_base", target_base)

    if source_base_array.shape != source_shape_array.shape:
        raise BlendShapeValidationError(
            "source_base and source_shape must have identical vertex counts/order"
        )
    if source_base_array.shape != target_base_array.shape:
        raise BlendShapeValidationError(
            "target_base must have the same vertex count as the source meshes"
        )

    topology_hash = None
    if (source_faces is None) != (target_faces is None):
        raise BlendShapeValidationError(
            "source_faces and target_faces must be provided together"
        )
    if source_faces is not None:
        source_faces_array = _faces("source_faces", source_faces)
        target_faces_array = _faces("target_faces", target_faces)
        if source_faces_array.shape != target_faces_array.shape or not np.array_equal(
            source_faces_array, target_faces_array
        ):
            raise BlendShapeValidationError(
                "source and target face topology/order are not identical"
            )
        max_index = int(source_faces_array.max())
        if max_index >= source_base_array.shape[0]:
            raise BlendShapeValidationError(
                "face index exceeds the available vertex count"
            )
        topology_hash = topology_sha256(source_faces_array)
    elif not confirmed_same_topology:
        raise BlendShapeValidationError(
            "face topology is absent; set confirmed_same_topology=True only after "
            "independently verifying vertex and face order"
        )

    offsets = source_shape_array - source_base_array
    displacement = np.linalg.norm(offsets, axis=1)
    maximum = float(displacement.max(initial=0.0))
    mean = float(displacement.mean())

    if max_displacement is not None:
        try:
            threshold = float(max_displacement)
        except (TypeError, ValueError) as exc:
            raise BlendShapeValidationError("max_displacement must be numeric") from exc
        if not math.isfinite(threshold) or threshold <= 0:
            raise BlendShapeValidationError("max_displacement must be finite and positive")
        if maximum > threshold:
            raise BlendShapeValidationError(
                f"maximum displacement {maximum:.8g} exceeds threshold {threshold:.8g}"
            )

    target_shape = target_base_array + offsets
    if not np.isfinite(target_shape).all():
        raise BlendShapeValidationError("transferred target shape is not finite")

    metadata = {
        "method": "same_topology_vertex_offset_transfer",
        "machine_learning": False,
        "topology_independent": False,
        "vertex_count": int(source_base_array.shape[0]),
        "face_topology_verified": source_faces is not None,
        "topology_sha256": topology_hash,
        "maximum_displacement": maximum,
        "mean_displacement": mean,
        "coordinate_unit": "input_defined",
    }
    return target_shape, offsets, metadata


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise BlendShapeValidationError(f"cannot read NPZ input: {exc}") from exc


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transfer a BlendShape only between explicitly identical topologies"
    )
    parser.add_argument("input", type=Path, help="NPZ with source_base/source_shape/target_base")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--confirmed-same-topology",
        action="store_true",
        help="explicit human confirmation; still requires matching face arrays when present",
    )
    parser.add_argument("--max-displacement", type=float)
    args = parser.parse_args()

    if not args.confirmed_same_topology:
        parser.error("--confirmed-same-topology is required")

    payload = _load_npz(args.input)
    required = {"source_base", "source_shape", "target_base"}
    missing = sorted(required - set(payload))
    if missing:
        raise BlendShapeValidationError(f"missing NPZ arrays: {', '.join(missing)}")

    source_faces = payload.get("source_faces")
    target_faces = payload.get("target_faces")
    target_shape, offsets, metadata = transfer_same_topology(
        payload["source_base"],
        payload["source_shape"],
        payload["target_base"],
        source_faces=source_faces,
        target_faces=target_faces,
        confirmed_same_topology=True,
        max_displacement=args.max_displacement,
    )
    metadata.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": str(args.input),
        }
    )
    output_arrays: dict[str, Any] = {
        "target_shape": target_shape,
        "offsets": offsets,
    }
    if target_faces is not None:
        output_arrays["target_faces"] = target_faces
    _atomic_write_npz(args.output, **output_arrays)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
