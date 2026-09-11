import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.blendshape_transfer import (
    BlendShapeValidationError,
    main,
    transfer_same_topology,
    verify_output_pair,
)


class BlendShapeTransferTests(unittest.TestCase):
    def setUp(self):
        self.source_base = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.source_shape = self.source_base + np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.2, 0.0], [0.1, 0.0, 0.0]]
        )
        self.target_base = self.source_base + 10.0
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)

    def test_offsets_are_transferred_deterministically(self):
        target, offsets, metadata = transfer_same_topology(
            self.source_base,
            self.source_shape,
            self.target_base,
            source_faces=self.faces,
            target_faces=self.faces.copy(),
        )
        np.testing.assert_allclose(offsets, self.source_shape - self.source_base)
        np.testing.assert_allclose(target, self.target_base + offsets)
        self.assertTrue(metadata["face_topology_verified"])
        self.assertFalse(metadata["machine_learning"])

    def test_different_face_order_is_rejected(self):
        with self.assertRaisesRegex(
            BlendShapeValidationError, "topology/order are not identical"
        ):
            transfer_same_topology(
                self.source_base,
                self.source_shape,
                self.target_base,
                source_faces=self.faces,
                target_faces=np.array([[0, 2, 1]]),
            )

    def test_missing_topology_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(
            BlendShapeValidationError, "confirmed_same_topology"
        ):
            transfer_same_topology(
                self.source_base, self.source_shape, self.target_base
            )

    def test_excessive_displacement_is_rejected(self):
        with self.assertRaisesRegex(BlendShapeValidationError, "exceeds threshold"):
            transfer_same_topology(
                self.source_base,
                self.source_shape,
                self.target_base,
                source_faces=self.faces,
                target_faces=self.faces,
                max_displacement=0.05,
            )

    def test_non_finite_vertices_are_rejected(self):
        invalid = self.source_base.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(BlendShapeValidationError, "NaN"):
            transfer_same_topology(
                invalid,
                self.source_shape,
                self.target_base,
                source_faces=self.faces,
                target_faces=self.faces,
            )


class OutputPairTests(unittest.TestCase):
    def _write_input(self, path: Path, *, target_offset: float = 10.0) -> None:
        source_base = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        source_shape = source_base + np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.2, 0.0], [0.1, 0.0, 0.0]]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        np.savez_compressed(
            path,
            source_base=source_base,
            source_shape=source_shape,
            target_base=source_base + target_offset,
            source_faces=faces,
            target_faces=faces,
        )

    def _run_cli(self, input_path: Path, output_path: Path) -> int:
        argv = [
            "blendshape_transfer.py",
            str(input_path),
            str(output_path),
            "--confirmed-same-topology",
        ]
        with patch.object(sys, "argv", argv):
            return main()

    def test_cli_publishes_bound_pair_and_rerun_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            self._write_input(input_path)

            self.assertEqual(self._run_cli(input_path, output_path), 0)
            first_metadata = verify_output_pair(output_path)
            with np.load(output_path, allow_pickle=False) as archive:
                first_target = archive["target_shape"].copy()
                first_offsets = archive["offsets"].copy()
            self.assertEqual(
                first_metadata["output_npz_sha256"],
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            )

            self.assertEqual(self._run_cli(input_path, output_path), 0)
            second_metadata = verify_output_pair(output_path)
            with np.load(output_path, allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["target_shape"], first_target)
                np.testing.assert_allclose(archive["offsets"], first_offsets)
            self.assertEqual(
                second_metadata["output_npz_sha256"],
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            )

    def test_metadata_finalization_failure_does_not_publish_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            self._write_input(input_path)

            with patch(
                "src.blendshape_transfer.Path.replace",
                side_effect=OSError("injected metadata finalization failure"),
            ):
                with self.assertRaisesRegex(OSError, "metadata finalization"):
                    self._run_cli(input_path, output_path)

            self.assertFalse(output_path.exists())
            self.assertFalse(output_path.with_suffix(".npz.json").exists())
            self.assertFalse(output_path.with_name("output.npz.pair.tmp").exists())

    def test_foreign_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_a = root / "input-a.npz"
            input_b = root / "input-b.npz"
            output_a = root / "output-a.npz"
            output_b = root / "output-b.npz"
            self._write_input(input_a, target_offset=10.0)
            self._write_input(input_b, target_offset=20.0)
            self._run_cli(input_a, output_a)
            self._run_cli(input_b, output_b)

            metadata_a = json.loads(
                output_a.with_suffix(".npz.json").read_text(encoding="utf-8")
            )
            output_b.with_suffix(".npz.json").write_text(
                json.dumps(metadata_a), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                BlendShapeValidationError, "does not match metadata"
            ):
                verify_output_pair(output_b)


if __name__ == "__main__":
    unittest.main()
