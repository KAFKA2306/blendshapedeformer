import unittest

import numpy as np

from src.blendshape_transfer import (
    BlendShapeValidationError,
    transfer_same_topology,
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


if __name__ == "__main__":
    unittest.main()
