"""Verified baseline utilities for blend-shape transfer."""

from .blendshape_transfer import (
    BlendShapeValidationError,
    transfer_same_topology,
)

__all__ = ["BlendShapeValidationError", "transfer_same_topology"]
