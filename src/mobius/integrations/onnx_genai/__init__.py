# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""onnx-genai integration for inference metadata generation."""

from mobius.integrations.onnx_genai.inference_metadata import (
    generate_inference_metadata,
    write_inference_metadata,
)

__all__ = ["generate_inference_metadata", "write_inference_metadata"]
