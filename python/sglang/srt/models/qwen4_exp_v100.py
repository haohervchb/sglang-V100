"""V100-specific Qwen4-Exp PLE host-offload initialization fix.

SGLang's model registry imports every module in this package. Importing this
module patches Qwen4ExpNGramEmbedding.__init__ so host-offloaded PLE tables are
created as metadata on the meta device and materialized directly in pinned host
memory by Qwen4ExpPinnedHostEmbedding.
"""

from sglang.srt.models import qwen4_exp as _base


def _v100_ngram_embedding_init(
    self,
    config,
    embedding_dim,
    ple_layer_index=0,
    quant_config=None,
):
    _base.nn.Module.__init__(self)
    self.config = config
    self.ngram_embed_dim = int(embedding_dim)
    self.ngram_size = int(config.ngram_size)
    self.heads_per_ngram = int(config.heads_per_ngram)
    self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
    self.ple_layer_index = int(ple_layer_index)
    self.unigram_vocab_size = int(config.vocab_size)
    if self.ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
    if self.heads_per_ngram <= 0:
        raise ValueError(
            f"heads_per_ngram must be > 0, got {self.heads_per_ngram}"
        )
    if self.ngram_embed_dim % self.ngram_heads != 0:
        raise ValueError(
            "ple_embed_dim must be divisible by total ngram heads: "
            f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
        )
    self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
    if self.ngram_vocab_size_base <= 0:
        raise ValueError("ngram_vocab_size_base must be > 0")
    self.make_ngram_vocab_size_divisible_by = int(
        config.make_ngram_vocab_size_divisible_by
    )
    self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
    self.eos_token_id = int(config.eos_token_id)
    self.enable_ple_fusion = _base.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

    self.register_buffer(
        "layer_multipliers",
        self._build_layer_multipliers(self.ngram_size),
        persistent=True,
    )
    head_vocab_sizes, head_offsets, total_vocab_size = (
        self._build_head_vocab_and_offsets()
    )
    self.register_buffer(
        "ngram_heads_vocab_sizes",
        _base.torch.tensor(head_vocab_sizes, dtype=_base.torch.long),
        persistent=True,
    )
    self.register_buffer(
        "ngram_heads_offsets",
        _base.torch.tensor(head_offsets, dtype=_base.torch.long),
        persistent=True,
    )
    padded_vocab_size = (
        (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
        // self.make_ngram_vocab_size_divisible_by
    ) * self.make_ngram_vocab_size_divisible_by
    self.use_attn_tp_ngram = _base._use_attn_tp_ngram()
    self.gather_dp_tokens = (
        _base.is_dp_attention_enabled()
        and _base.get_attention_dp_size() > 1
        and not self.use_attn_tp_ngram
    )

    init_context = (
        _base.torch.device("meta")
        if config.ple_offload_embedding
        else _base.nullcontext()
    )
    with init_context:
        self.ngram_embedding = _base.VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim_per_ngram,
            params_dtype=(
                _base.torch.float8_e4m3fn
                if (
                    quant_config is not None
                    and quant_config.get_name() == "fp8"
                )
                or getattr(config, "ple_embedding_dtype", None)
                == "float8_e4m3fn"
                else _base.torch.bfloat16
            ),
            output_dtype=_base.torch.get_default_dtype(),
            use_attn_tp_group=self.use_attn_tp_ngram,
        )
    self.ngram_embedding.register_buffer(
        "weight_scale",
        _base.torch.ones(1, dtype=_base.torch.get_default_dtype()),
        persistent=True,
    )


_base.Qwen4ExpNGramEmbedding.__init__ = _v100_ngram_embedding_init
