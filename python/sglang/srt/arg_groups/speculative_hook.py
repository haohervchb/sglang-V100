import json
import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_BUILTIN_MTP_MODEL_ARCHS = {
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "GlmMoeDsaForCausalLM",
    "BailingMoeForCausalLM",
    "BailingMoeV2ForCausalLM",
    "BailingMoeV2_5ForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "HYV3ForCausalLM",
}

_OPTIONAL_EXTERNAL_MTP_MODEL_ARCHS = {
    "MistralLarge3ForCausalLM",
    "PixtralForConditionalGeneration",
}


def _resolve_speculative_algorithm_alias(
    speculative_algorithm: Optional[str],
    speculative_draft_model_path: Optional[str],
    trust_remote_code: bool = False,
    kwargs: Optional[dict] = {},
) -> Optional[str]:
    """Resolve CLI speculative algorithm; NEXTN/EAGLE may become FROZEN_KV_MTP for Gemma4 assistant drafts."""

    is_gemma4_draft = False
    if speculative_draft_model_path:
        from sglang.srt.utils.hf_transformers_utils import get_config

        cfg = get_config(
            speculative_draft_model_path, trust_remote_code=trust_remote_code, **kwargs
        )
        is_gemma4_draft = "Gemma4AssistantForCausalLM" in (
            getattr(cfg, "architectures", None) or []
        )

    if speculative_algorithm == "EAGLE3" and is_gemma4_draft:
        raise ValueError(
            "Gemma4AssistantForCausalLM draft requires "
            "--speculative-algorithm NEXTN or EAGLE; EAGLE3 is "
            "not supported for this draft architecture."
        )

    if speculative_algorithm == "NEXTN" or speculative_algorithm == "EAGLE":
        if is_gemma4_draft:
            logger.info(
                "Detected Gemma4AssistantForCausalLM draft; "
                f"promoting --speculative-algorithm {speculative_algorithm} to FROZEN_KV_MTP."
            )
            return "FROZEN_KV_MTP"
        return "EAGLE"

    return speculative_algorithm


def handle_speculative_decoding(server_args: "ServerArgs") -> None:
    if (
        server_args.speculative_draft_model_path is not None
        and server_args.speculative_draft_model_revision is None
    ):
        server_args.speculative_draft_model_revision = "main"

    if server_args.speculative_moe_runner_backend is None:
        server_args.speculative_moe_runner_backend = server_args.moe_runner_backend

    if server_args.speculative_algorithm is not None:
        server_args.speculative_algorithm = server_args.speculative_algorithm.upper()

    kwargs = {}

    override_config_file = server_args.decrypted_draft_config_file
    if override_config_file and override_config_file.strip():
        kwargs["_configuration_file"] = override_config_file.strip()

    server_args.speculative_algorithm = _resolve_speculative_algorithm_alias(
        server_args.speculative_algorithm,
        server_args.speculative_draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
        kwargs=kwargs,
    )

    # Validate --speculative-draft-window-size once, regardless of algorithm.
    # Consumed by DFLASH (compact draft KV cache) and Llama EAGLE-3 (drafter attention SWA).
    if server_args.speculative_draft_window_size is not None:
        window_size = int(server_args.speculative_draft_window_size)
        if window_size <= 0:
            raise ValueError(
                f"--speculative-draft-window-size must be positive, got {window_size}."
            )
        server_args.speculative_draft_window_size = window_size
        if server_args.speculative_algorithm not in ("EAGLE3", "DFLASH", "DSPARK"):
            logger.warning(
                "--speculative-draft-window-size has no effect with "
                "speculative_algorithm=%s (honored by Llama EAGLE-3, DFLASH, and DSPARK only).",
                server_args.speculative_algorithm,
            )

    if server_args.speculative_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo

        algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        # TODO: move the per-algorithm validation below into spec module hooks.
        if isinstance(algo, CustomSpecAlgo) and algo.validate_server_args is not None:
            algo.validate_server_args(server_args)

    if server_args.speculative_skip_dp_mlp_sync:
        assert server_args.speculative_algorithm == "EAGLE", (
            "--speculative-skip-dp-mlp-sync is only supported with "
            f"speculative_algorithm == EAGLE, got {server_args.speculative_algorithm}."
        )

    if server_args.speculative_algorithm == "DFLASH":
        _handle_dflash(server_args)
    elif server_args.speculative_algorithm == "DSPARK":
        _handle_dspark(server_args)
    elif server_args.speculative_algorithm == "FROZEN_KV_MTP":
        _handle_frozen_kv_mtp(server_args)
    elif server_args.speculative_algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
        _handle_eagle_family(server_args)
    elif server_args.speculative_algorithm == "NGRAM":
        _handle_ngram(server_args)

    if server_args.speculative_adaptive:
        _maybe_disable_adaptive(server_args)


def _handle_dflash(server_args: "ServerArgs") -> None:
    if server_args.enable_dp_attention:
        raise ValueError(
            "Currently DFLASH speculative decoding does not support dp attention."
        )

    if server_args.pp_size != 1:
        raise ValueError(
            "Currently DFLASH speculative decoding only supports pp_size == 1."
        )

    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "DFLASH speculative decoding requires setting --speculative-draft-model-path."
        )

    # ServerArgs normally inherits the target quantization for a draft model.
    # DFlash checkpoints are independent models and may use a different format
    # (for example, an unquantized DFlash draft with a GPTQ target).
    draft_hf_config = None
    try:
        from sglang.srt.utils.hf_transformers_utils import get_config

        draft_hf_config = get_config(
            server_args.speculative_draft_model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.speculative_draft_model_revision,
            model_override_args=json.loads(server_args.json_model_override_args),
        )
    except Exception as e:
        logger.warning(
            "Failed to inspect the DFLASH draft config before model loading: %s", e
        )

    draft_architectures = (
        getattr(draft_hf_config, "architectures", None)
        if draft_hf_config is not None
        else None
    )
    is_laguna_dflash_on_v100 = (
        draft_architectures is not None
        and "DFlashLagunaForCausalLM" in draft_architectures
        and getattr(server_args, "attention_backend", None)
        in ("tilelang_fa_v100", "flash_attn_v100")
    )

    if (
        draft_hf_config is not None
        and getattr(draft_hf_config, "quantization_config", None) is None
        and server_args.speculative_draft_model_quantization is not None
    ):
        logger.info(
            "The DFLASH draft checkpoint is unquantized; ignoring inherited "
            "draft quantization %r from the target model.",
            server_args.speculative_draft_model_quantization,
        )
        server_args.speculative_draft_model_quantization = None

    # DFLASH does not use EAGLE-style `num_steps`/`topk`, but those fields still
    # affect generic scheduler/KV-cache accounting (buffer sizing, KV freeing,
    # RoPE reservation). Force them to 1 to avoid surprising memory behavior.
    #
    # For DFlash, the natural unit is `block_size` (verify window length).
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DFLASH only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DFLASH only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    if server_args.speculative_dflash_block_size is not None:
        if int(server_args.speculative_dflash_block_size) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {server_args.speculative_dflash_block_size}."
            )
        if server_args.speculative_num_draft_tokens is not None and int(
            server_args.speculative_num_draft_tokens
        ) != int(server_args.speculative_dflash_block_size):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. For DFLASH they must match. "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
                f"speculative_dflash_block_size={server_args.speculative_dflash_block_size}."
            )
        server_args.speculative_num_draft_tokens = int(
            server_args.speculative_dflash_block_size
        )

    if (
        server_args.speculative_num_draft_tokens is None
        and is_laguna_dflash_on_v100
    ):
        # The checkpoint stores the block-16 throughput configuration, while
        # Poolside recommends seven speculative tokens (block size 8) for
        # general serving. Use that neutral default on SM70. Block 16 remains
        # valuable for high-acceptance math/code workloads, while block 4 is a
        # defensive option for long-form prose and repository review.
        # Explicit CLI values above remain authoritative.
        server_args.speculative_num_draft_tokens = 8
        logger.info(
            "Using Poolside's recommended DFLASH block size 8 for Laguna with "
            "the V100 TileLang backend. "
            "Override with --speculative-dflash-block-size after benchmarking "
            "acceptance on your workload."
        )

    if server_args.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import (
            parse_dflash_draft_config,
        )

        inferred_block_size = None
        try:
            if draft_hf_config is not None:
                inferred_block_size = parse_dflash_draft_config(
                    draft_hf_config=draft_hf_config
                ).resolve_block_size(default=None)
        except Exception as e:
            logger.warning(
                "Failed to infer DFLASH block_size from draft model config; "
                "defaulting speculative_num_draft_tokens to 16. Error: %s",
                e,
            )

        if inferred_block_size is None:
            inferred_block_size = 16
            logger.warning(
                "speculative_num_draft_tokens is not set; defaulting to %d for DFLASH.",
                inferred_block_size,
            )
        server_args.speculative_num_draft_tokens = inferred_block_size

    if server_args.speculative_draft_window_size is not None:
        draft_tokens = int(server_args.speculative_num_draft_tokens)
        if server_args.speculative_draft_window_size < draft_tokens:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-num-draft-tokens (block_size). "
                f"window_size={server_args.speculative_draft_window_size}, block_size={draft_tokens}."
            )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if not envs.SGLANG_ENABLE_SPEC_V2.get():
        # The V1 worker only supports non-overlap scheduling.
        server_args.disable_overlap_schedule = True
        logger.warning(
            "Spec v1 is used for DFLASH speculative decoding because "
            "SGLANG_ENABLE_SPEC_V2 is off; overlap schedule is disabled."
        )
    else:
        logger.warning("Spec v2 is enabled by default for DFLASH speculative decoding.")

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dflash speculative decoding."
        )


def _handle_dspark(server_args: "ServerArgs") -> None:
    """Normalize the fixed-width DSpark contract used by the SM70 worker.

    DSpark's checkpoint ``block_size`` is gamma (the number of proposed
    tokens), while SGLang's shared speculative accounting uses the target
    verification width.  The latter is therefore gamma + 1 for the anchor.
    """
    if server_args.enable_dp_attention:
        raise ValueError("DSPARK speculative decoding does not support dp attention yet.")
    if server_args.pp_size != 1:
        raise ValueError("DSPARK speculative decoding only supports pp_size == 1.")
    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "DSPARK speculative decoding requires --speculative-draft-model-path."
        )

    draft_hf_config = None
    try:
        from sglang.srt.utils.hf_transformers_utils import get_config

        draft_hf_config = get_config(
            server_args.speculative_draft_model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.speculative_draft_model_revision,
            model_override_args=json.loads(server_args.json_model_override_args),
        )
    except Exception as e:
        logger.warning("Failed to inspect the DSPARK draft config: %s", e)

    if draft_hf_config is not None:
        architectures = getattr(draft_hf_config, "architectures", None) or []
        if "DSparkDraftModel" not in architectures:
            raise ValueError(
                "DSPARK requires a DSparkDraftModel checkpoint, but got "
                f"architectures={architectures}."
            )
        if (
            getattr(draft_hf_config, "quantization_config", None) is None
            and server_args.speculative_draft_model_quantization is not None
        ):
            logger.info(
                "The DSPARK draft is unquantized; ignoring inherited draft quantization %r.",
                server_args.speculative_draft_model_quantization,
            )
            server_args.speculative_draft_model_quantization = None

    server_args.speculative_num_steps = 1
    server_args.speculative_eagle_topk = 1

    gamma = server_args.speculative_dspark_block_size
    if gamma is None and draft_hf_config is not None:
        gamma = getattr(draft_hf_config, "block_size", None)
    if gamma is None:
        gamma = 7
        logger.warning("DSPARK block size is not set; defaulting gamma to 7.")
    gamma = int(gamma)
    if gamma <= 0:
        raise ValueError(
            "DSPARK requires --speculative-dspark-block-size to be positive, "
            f"got {gamma}."
        )

    verify_width = gamma + 1
    if (
        server_args.speculative_num_draft_tokens is not None
        and int(server_args.speculative_num_draft_tokens) != verify_width
    ):
        raise ValueError(
            "DSPARK --speculative-num-draft-tokens is the verify width and must "
            f"equal gamma + 1 ({verify_width}), got "
            f"{server_args.speculative_num_draft_tokens}."
        )
    server_args.speculative_dspark_block_size = gamma
    server_args.speculative_num_draft_tokens = verify_width

    if (
        server_args.speculative_draft_window_size is not None
        and int(server_args.speculative_draft_window_size) < verify_width
    ):
        raise ValueError(
            "--speculative-draft-window-size must be at least the DSPARK "
            f"verify width ({verify_width})."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning("Mixed chunked prefill is disabled for DSPARK.")


def _handle_frozen_kv_mtp(server_args: "ServerArgs") -> None:
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.disable_overlap_schedule = True
    logger.warning(
        "Overlap scheduler is disabled when using Frozen-KV MTP speculative decoding (spec v2 is not supported yet)."
    )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "Frozen-KV MTP speculative decoding."
        )


def _handle_eagle_family(server_args: "ServerArgs") -> None:
    if (
        server_args.speculative_algorithm == "STANDALONE"
        and server_args.enable_dp_attention
    ):
        # TODO: support dp attention for standalone speculative decoding
        raise ValueError(
            "Currently standalone speculative decoding does not support dp attention."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    spec_v1_reason = None
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk > 1
        and not server_args.disable_overlap_schedule
    ):
        server_args.disable_overlap_schedule = True
        spec_v1_reason = "spec v2 currently only supports topk = 1"
    elif (
        not envs.SGLANG_ENABLE_SPEC_V2.get()
        and not server_args.disable_overlap_schedule
    ):
        server_args.disable_overlap_schedule = True
        spec_v1_reason = "SGLANG_ENABLE_SPEC_V2=False"

    if server_args.disable_overlap_schedule:
        logger.warning(
            "Spec v1 is used for eagle/eagle3/standalone speculative decoding because %s.",
            spec_v1_reason or "overlap schedule is disabled",
        )
    else:
        logger.warning(
            "Spec v2 is enabled by default for eagle/eagle3/standalone speculative decoding."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "eagle speculative decoding."
        )

    model_arch = server_args.get_model_config().hf_config.architectures[0]
    if model_arch in (
        _BUILTIN_MTP_MODEL_ARCHS | _OPTIONAL_EXTERNAL_MTP_MODEL_ARCHS
    ) or model_arch == "Qwen4ExpForConditionalGeneration":
        if server_args.speculative_draft_model_path is None:
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
        elif model_arch in _BUILTIN_MTP_MODEL_ARCHS:
            logger.warning(
                "%s has a built-in MTP layer and does not require setting "
                "speculative_draft_model_path.",
                model_arch,
            )

    if server_args.speculative_num_steps is None:
        assert (
            server_args.speculative_eagle_topk is None
            and server_args.speculative_num_draft_tokens is None
        )
        from sglang.srt.server_args import auto_choose_speculative_params

        (
            server_args.speculative_num_steps,
            server_args.speculative_eagle_topk,
            server_args.speculative_num_draft_tokens,
        ) = auto_choose_speculative_params(server_args)

    if (
        server_args.attention_backend == "trtllm_mha"
        or server_args.decode_attention_backend == "trtllm_mha"
        or server_args.prefill_attention_backend == "trtllm_mha"
    ):
        if server_args.speculative_eagle_topk > 1:
            raise ValueError(
                "trtllm_mha backend only supports topk = 1 for speculative decoding."
            )

    if (
        server_args.speculative_eagle_topk == 1
        and server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
        server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend not in ["flashinfer", "fa3"]
    ):
        raise ValueError(
            "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
        )


def _handle_ngram(server_args: "ServerArgs") -> None:
    if not server_args.device.startswith("cuda"):
        raise ValueError("Ngram speculative decoding only supports CUDA device.")

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.disable_overlap_schedule = True
    server_args.enable_mixed_chunk = False
    server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 12
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if server_args.speculative_ngram_external_corpus_path is not None:
        if server_args.speculative_ngram_external_sam_budget <= 0:
            raise ValueError(
                "--speculative-ngram-external-sam-budget must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if server_args.speculative_ngram_external_corpus_max_tokens <= 0:
            raise ValueError(
                "--speculative-ngram-external-corpus-max-tokens must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if (
            server_args.speculative_ngram_external_sam_budget
            > server_args.speculative_num_draft_tokens - 1
        ):
            raise ValueError(
                "speculative_ngram_external_sam_budget must be less than or equal to "
                f"speculative_num_draft_tokens - 1 ({server_args.speculative_num_draft_tokens - 1})."
            )
    logger.warning(
        "The overlap scheduler and mixed chunked prefill are disabled because of "
        "using ngram speculative decoding."
    )

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({server_args.speculative_eagle_topk}) > 1 "
            f"with page_size({server_args.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if server_args.enable_dp_attention:
        # TODO: support dp attention for ngram speculative decoding
        raise ValueError(
            "Currently ngram speculative decoding does not support dp attention."
        )


def _maybe_disable_adaptive(server_args: "ServerArgs") -> None:
    from sglang.srt.speculative.adaptive_spec_params import (
        adaptive_unsupported_reason,
    )

    reason = adaptive_unsupported_reason(server_args)
    if reason is not None:
        logger.warning(
            f"speculative_adaptive disabled: {reason}. "
            "Falling back to static speculative params."
        )
        server_args.speculative_adaptive = False
