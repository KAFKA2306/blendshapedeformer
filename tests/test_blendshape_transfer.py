import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import blendshape_transfer
from src.blendshape_transfer import (
    BlendShapeValidationError,
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

    def _write_input(self, path: Path) -> None:
        np.savez_compressed(
            path,
            source_base=self.source_base,
            source_shape=self.source_shape,
            target_base=self.target_base,
            source_faces=self.faces,
            target_faces=self.faces.copy(),
        )

    def _run_cli(self, input_path: Path, output_path: Path) -> int:
        argv = [
            "blendshape_transfer.py",
            str(input_path),
            str(output_path),
            "--confirmed-same-topology",
        ]
        with mock.patch.object(sys, "argv", argv):
            return blendshape_transfer.main()

    def test_cli_publishes_verified_npz_metadata_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.npz"
            first_output = root / "first.npz"
            second_output = root / "second.npz"
            self._write_input(input_path)

            self.assertEqual(self._run_cli(input_path, first_output), 0)
            first_metadata = verify_output_pair(first_output)
            with np.load(first_output, allow_pickle=False) as first:
                first_target = first["target_shape"].copy()
                first_offsets = first["offsets"].copy()

            self.assertEqual(self._run_cli(input_path, second_output), 0)
            second_metadata = verify_output_pair(second_output)
            with np.load(second_output, allow_pickle=False) as second:
                np.testing.assert_allclose(first_target, second["target_shape"])
                np.testing.assert_allclose(first_offsets, second["offsets"])

            self.assertEqual(first_metadata["vertex_count"], second_metadata["vertex_count"])
            for output, metadata in (
                (first_output, first_metadata),
                (second_output, second_metadata),
            ):
                self.assertEqual(metadata["output_npz_size_bytes"], output.stat().st_size)
                self.assertEqual(len(metadata["output_npz_sha256"]), 64)

    def test_metadata_finalization_failure_does_not_publish_npz(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            metadata_path = output_path.with_suffix(".npz.json")
            self._write_input(input_path)
            real_replace = blendshape_transfer._replace_file

            def fail_metadata(source: Path, target: Path) -> None:
                if target == metadata_path:
                    raise OSError("injected metadata finalization failure")
                real_replace(source, target)

            with mock.patch.object(
                blendshape_transfer, "_replace_file", side_effect=fail_metadata
            ):
                with self.assertRaisesRegex(OSError, "metadata finalization"):
                    self._run_cli(input_path, output_path)

            self.assertFalse(output_path.exists())
            self.assertFalse(metadata_path.exists())
            self.assertFalse(output_path.with_name("output.npz.tmp").exists())
            self.assertFalse(metadata_path.with_name("output.npz.json.tmp").exists())

    def test_verify_output_pair_rejects_foreign_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            self._write_input(input_path)
            self.assertEqual(self._run_cli(input_path, output_path), 0)

            metadata_path = output_path.with_suffix(".npz.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["output_npz_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(BlendShapeValidationError, "SHA-256"):
                verify_output_pair(output_path)


if __name__ == "__main__":
    unittest.main()
