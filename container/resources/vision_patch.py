"""Replaces the vision tower's Conv3d patch embedding with the equivalent matmul.

Qwen3_5VisionPatchEmbed runs a Conv3d with kernel == stride over a batch of one
tiny 2x16x16 volume per patch, so each patch yields exactly one output and the op
is a plain (N, 1536) @ (1536, 1152) matmul in disguise. PyTorch's bf16 Conv3d
path degenerates at our batch sizes: measured 1,487.6 s for a 768-frame sample
(337,920 patches) against 5 ms for the same matmul -- 99% of the vision tower's
entire forward. Weights are identical; only the op changes.

Call `apply()` before the first forward; the trainer and evaluator both do.
"""

from __future__ import annotations

import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

_original_forward = m.Qwen3_5VisionPatchEmbed.forward


def _linear_forward(self, hidden_states):
    weight = self.proj.weight
    return F.linear(hidden_states.to(weight.dtype),
                    weight.reshape(weight.shape[0], -1), self.proj.bias)


def apply() -> None:
    m.Qwen3_5VisionPatchEmbed.forward = _linear_forward


def remove() -> None:
    m.Qwen3_5VisionPatchEmbed.forward = _original_forward
