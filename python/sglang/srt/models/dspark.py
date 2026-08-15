from __future__ import annotations

from typing import Iterable, Tuple

import torch
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dspark_config import parse_dspark_draft_config
from torch import nn


class VanillaMarkov(nn.Module):
    """The trained low-rank autoregressive correction used by dense DSpark."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(int(vocab_size), int(rank))
        self.markov_w2 = nn.Linear(int(rank), int(vocab_size), bias=False)

    def step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids.long()))


class DSparkDraftModel(DFlashDraftModel):
    """Dense DSpark draft backbone with its trained Markov correction head.

    The target owns the embeddings and vocabulary head.  Keeping those shared
    avoids another copy of the 248k-token Qwen vocabulary weights.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        dspark = parse_dspark_draft_config(config)
        self.gamma = dspark.gamma
        self.markov_head = VanillaMarkov(
            vocab_size=int(config.vocab_size), rank=dspark.markov_rank
        )

    @torch.no_grad()
    def sample_greedy_block(
        self,
        *,
        hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
        lm_head: nn.Module,
    ) -> torch.Tensor:
        """Generate gamma tokens with the checkpoint's semi-AR recurrence."""
        if hidden_states.ndim != 3 or int(hidden_states.shape[1]) != self.gamma:
            raise ValueError(
                "DSPARK draft hidden shape mismatch: expected "
                f"[batch, {self.gamma}, hidden], got {tuple(hidden_states.shape)}."
            )
        if not hasattr(lm_head, "weight"):
            raise ValueError("DSPARK requires the target vocabulary-parallel lm_head.")

        weight = lm_head.weight
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat_hidden.dtype != weight.dtype:
            flat_hidden = flat_hidden.to(weight.dtype)
        local_logits = torch.matmul(flat_hidden, weight.T)
        logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        vocab_size = int(self.config.vocab_size)
        logits = logits[..., :vocab_size].view(
            hidden_states.shape[0], self.gamma, vocab_size
        )

        proposed = []
        previous = anchor_tokens.long()
        for step in range(self.gamma):
            step_logits = logits[:, step, :] + self.markov_head.step_bias(previous)
            previous = torch.argmax(step_logits, dim=-1)
            proposed.append(previous)
        return torch.stack(proposed, dim=1)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        backbone_weights = []
        markov_weights = []
        for name, loaded_weight in weights:
            if name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            elif name.startswith(("confidence_head.", "embed_tokens.", "lm_head.")):
                # Static DSpark verification does not consume the confidence head;
                # embeddings/lm_head are shared from the target model.
                continue
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)
        params = dict(self.named_parameters())
        loaded_names = set()
        for name, loaded_weight in markov_weights:
            if name not in params:
                raise ValueError(f"Unexpected DSPARK Markov weight {name!r}.")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_names.add(name)

        expected = {
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        }
        missing = expected - loaded_names
        if missing:
            raise ValueError(
                f"DSPARK checkpoint is missing Markov weights: {sorted(missing)}"
            )


EntryClass = [DSparkDraftModel]
