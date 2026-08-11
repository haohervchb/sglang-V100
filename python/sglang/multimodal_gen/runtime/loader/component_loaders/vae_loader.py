import importlib.util
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)
VAE_CHANNELS_LAST_3D_ENV = "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D"


def _resolve_stream_state_key(name: str, state: dict[str, torch.Tensor]) -> str | None:
    """Resolve checkpoint aliases without materializing a second state dict.

    PyTorch's legacy ``torch.nn.utils.weight_norm`` serialized ``weight_g`` and
    ``weight_v``.  The parametrization API used by current native VAEs exposes
    the same tensors as ``parametrizations.weight.original0`` and
    ``parametrizations.weight.original1``.  MiniMax H3 ships the legacy names,
    so translate them while retaining strict shape/missing-key validation.
    """
    if name in state:
        return name

    suffix_aliases = {
        "weight_g": "parametrizations.weight.original0",
        "weight_v": "parametrizations.weight.original1",
    }
    for suffix, replacement in suffix_aliases.items():
        if name.endswith(suffix):
            candidate = f"{name[: -len(suffix)]}{replacement}"
            if candidate in state:
                return candidate
    return None


def _backfill_ltx2_audio_vae_latent_stats(
    loaded: dict[str, torch.Tensor], component_name: str
) -> None:
    if component_name != "audio_vae":
        return
    mean_key = "per_channel_statistics.mean-of-means"
    std_key = "per_channel_statistics.std-of-means"
    if "latents_mean" not in loaded and mean_key in loaded:
        loaded["latents_mean"] = loaded[mean_key]
    if "latents_std" not in loaded and std_key in loaded:
        loaded["latents_std"] = loaded[std_key]


@torch.no_grad()
def _stream_safetensors_into_module(
    module: nn.Module,
    safetensors_list: list[str],
) -> None:
    """Strictly load safetensors without retaining a full checkpoint copy."""
    state = module.state_dict()
    expected = set(state)
    loaded: set[str] = set()

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                state_name = _resolve_stream_state_key(name, state)
                if state_name is None:
                    raise RuntimeError(
                        f"Unexpected checkpoint key {name!r} while loading {path}"
                    )
                if state_name in loaded:
                    raise RuntimeError(f"Duplicate checkpoint key {name!r}")
                source = handle.get_tensor(name)
                target = state[state_name]
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name!r}: checkpoint "
                        f"{tuple(source.shape)} != model {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded.add(state_name)

    missing = expected - loaded
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {sorted(missing)}")


def _convert_conv3d_weights_to_channels_last_3d(module: nn.Module) -> int:
    """
    Convert Conv3d weights to channels_last_3d (NDHWC) memory format.
    Returns the number of Conv3d modules converted.
    """
    if not hasattr(torch, "channels_last_3d"):
        return 0
    num_converted = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            try:
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                num_converted += 1
            except Exception:
                # Best-effort; skip unsupported cases.
                continue
    return num_converted


def _should_use_channels_last_3d(
    server_args: ServerArgs | None, component_name: str
) -> bool:
    if component_name not in (
        "vae",
        "video_vae",
    ) or not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    override = os.getenv(VAE_CHANNELS_LAST_3D_ENV)
    if override is not None and override.strip().lower() != "auto":
        return get_bool_env_var(VAE_CHANNELS_LAST_3D_ENV)

    if server_args is None:
        return False

    pipeline_name = server_args.pipeline_config.__class__.__name__
    if pipeline_name.startswith("QwenImage"):
        return True
    if "Wan" in pipeline_name and server_args.num_gpus == 1:
        return True
    return False


class VAELoader(ComponentLoader):
    """Shared loader for (video/audio) VAE modules."""

    component_names = ["vae", "audio_vae", "video_vae"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the VAE based on the model path, and inference args."""
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert class_name is not None, (
            "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        )

        server_args.model_paths[component_name] = component_model_path

        if component_name in ("vae", "video_vae"):
            pipeline_vae_config_attr = "vae_config"
            pipeline_vae_precision = "vae_precision"
        elif component_name in ("audio_vae",):
            pipeline_vae_config_attr = "audio_vae_config"
            pipeline_vae_precision = "audio_vae_precision"
        else:
            raise ValueError(
                f"Unsupported module name for VAE loader: {component_name}"
            )
        vae_config = getattr(server_args.pipeline_config, pipeline_vae_config_attr)
        vae_precision = getattr(server_args.pipeline_config, pipeline_vae_precision)
        vae_config.update_model_arch(config)
        if hasattr(vae_config, "post_init"):
            # NOTE: some post init logics are only available after updated with config
            vae_config.post_init()

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        native_only = component_name in getattr(
            server_args.pipeline_config, "native_only_components", ()
        )
        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if auto_model_map and not native_only:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            vae_dtype = PRECISION_TO_TYPE[vae_precision]
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            if _should_use_channels_last_3d(server_args, component_name):
                n = _convert_conv3d_weights_to_channels_last_3d(vae)
                if n > 0:
                    logger.info(
                        "VAE: converted %d Conv3d weights to channels_last_3d", n
                    )
            vae = current_platform.optimize_vae(vae)
            return vae

        # Load from ModelRegistry (standard VAE classes)
        with (
            set_default_torch_dtype(PRECISION_TO_TYPE[vae_precision]),
            skip_init_modules(),
        ):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        safetensors_list = server_args.pipeline_config.select_vae_weight_files(
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            component_name=component_name,
            vae_precision=vae_precision,
        )

        assert len(safetensors_list) >= 1, (
            f"Found no safetensors files in {component_model_path}"
        )
        strict_load = native_only
        if strict_load:
            # H3's video VAE alone is ~9.7 GiB. Every TP worker owns a VAE,
            # so retaining a second full state dict per process can exhaust
            # host memory before online quantization has bought any headroom.
            _stream_safetensors_into_module(vae, safetensors_list)
        else:
            loaded = {}
            for sf_path in safetensors_list:
                loaded.update(safetensors_load_file(sf_path))
            _backfill_ltx2_audio_vae_latent_stats(loaded, component_name)
            vae.load_state_dict(loaded, strict=False)
            state_keys = set(vae.state_dict().keys())
            loaded_keys = set(loaded.keys())
            missing_keys = sorted(state_keys - loaded_keys)
            unexpected_keys = sorted(loaded_keys - state_keys)
            if missing_keys:
                logger.warning("VAE missing keys: %s", missing_keys)
            if unexpected_keys:
                logger.warning("VAE unexpected keys: %s", unexpected_keys)

        if _should_use_channels_last_3d(server_args, component_name):
            n = _convert_conv3d_weights_to_channels_last_3d(vae)
            if n > 0:
                logger.info("VAE: converted %d Conv3d weights to channels_last_3d", n)

        vae = current_platform.optimize_vae(vae)
        return vae
