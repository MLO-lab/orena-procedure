"""Compute logits only at supervised positions.

A procedure prompt is ~66,000 tokens and exactly ~10 of them are training targets --
everything before the assistant marker is masked to -100. The stock causal-LM path
still runs `lm_head` over the whole sequence and then upcasts to fp32 for the loss:

    66,300 x 248,320 vocab x 2 bytes (bf16)  = 32.9 GB
                             x 4 bytes (fp32) = 65.9 GB
                                        peak ~= 99 GB, for ~10 useful rows

That alone exceeds an 80 GB H100 before weights, activations or optimizer state. It is
also almost entirely waste: cross-entropy ignores every -100 position, so those logits
are computed and discarded.

`Qwen3_5ForConditionalGeneration.forward` accepts a tensor for `logits_to_keep` and
slices hidden states with it before `lm_head`, so passing the handful of positions the
loss actually reads turns 33 GB into a few megabytes. Nothing about the gradient
changes -- the discarded rows contributed exactly zero.

The arithmetic is asserted against a full-logits reference in
`tests/test_sparse_logits.py`, on random tensors, so the equivalence is checked rather
than argued.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from trl import SFTTrainer


def supervised_positions(labels: torch.Tensor) -> torch.Tensor:
    """Hidden-state indices whose logits the loss will read.

    A label at position `t` is predicted from the hidden state at `t-1`, so column `j`
    of `labels[:, 1:]` needs hidden index `j`. The union across the batch is taken
    because the slice is shared by every sample; positions a given sample does not care
    about arrive carrying -100 and are ignored by cross-entropy anyway.
    """
    return (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]


def sparse_causal_loss(model, inputs: dict, labels: torch.Tensor,
                       num_items_in_batch: int | None = None):
    keep = supervised_positions(labels)
    if keep.numel() == 0:                                  # nothing supervised in this batch
        zero = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
        return zero, None

    outputs = model(**inputs, logits_to_keep=keep)
    logits = outputs.logits                                 # (B, K, V), K = keep.numel()
    targets = labels[:, 1:].index_select(1, keep)           # (B, K)

    # Mirrors transformers.loss.loss_utils.fixed_cross_entropy: sum / num_items_in_batch
    # when the Trainer supplies it (which is what makes gradient accumulation match a
    # single large batch), mean otherwise.
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1)
    if num_items_in_batch is None:
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
    else:
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100, reduction="sum")
        loss = loss / num_items_in_batch
    return loss, outputs


class SparseLogitsSFTTrainer(SFTTrainer):
    """SFTTrainer that never materialises full-sequence logits.

    Drop-in: the only override is `compute_loss`. Evaluation goes through the same path,
    so `eval_loss` stays comparable to a run trained without it.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        loss, outputs = sparse_causal_loss(model, inputs, labels, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss
