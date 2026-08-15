import logging
from typing import Optional

import torch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_config import parse_dspark_draft_config

logger = logging.getLogger(__name__)


class DSparkWorkerV2(DFlashWorkerV2):
    """Fixed-width DSpark on the native SM70 DFlash execution machinery."""

    def _draft_kv_cache_dtype(self, server_args: ServerArgs) -> str:
        # V100 has no native FP8 arithmetic.  Keeping the small five-layer
        # drafter cache in FP16 avoids an FP8 dequantization pass on every
        # proposal while preserving the user's target-cache choice.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7:
            return "auto"
        return super()._draft_kv_cache_dtype(server_args)

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        config = parse_dspark_draft_config(
            self.draft_model_runner.model_config.hf_config
        )
        self.gamma = int(config.gamma)
        self.draft_forward_size = self.gamma
        if int(self.block_size) != self.gamma + 1:
            raise ValueError(
                "DSPARK target verify width must equal gamma + 1: "
                f"width={self.block_size}, gamma={self.gamma}."
            )
        if not hasattr(self.draft_model, "sample_greedy_block"):
            raise TypeError(
                "DSPARK loaded an incompatible draft model: "
                f"{type(self.draft_model).__name__}."
            )

        # The draft evaluates gamma positions; the target evaluates the anchor
        # plus gamma proposals.  They deliberately use different graph widths.
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=self.gamma,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DSPARK SM70 worker. gamma=%d, verify_width=%d, model=%s",
                self.gamma,
                self.block_size,
                type(self.draft_model).__name__,
            )

    def _populate_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        lm_head,
        draft_tokens: torch.Tensor,
    ) -> None:
        proposals = self.draft_model.sample_greedy_block(
            hidden_states=draft_hidden,
            anchor_tokens=block_ids[:, 0],
            lm_head=lm_head,
        )
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(proposals)
