"""Regression coverage for Laguna's regex-only INT4 expert scheme."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)


class TestLagunaCompressedTensors(unittest.TestCase):
    def test_regex_only_group_resolves_w4a16_moe(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": [
                            r"re:.*layers\.\d+\..*"
                            r"(w[1-3]|gate_proj|up_proj|down_proj)$"
                        ],
                        "weights": {
                            "dynamic": False,
                            "group_size": 32,
                            "num_bits": 4,
                            "strategy": "group",
                            "symmetric": True,
                            "type": "int",
                        },
                        "input_activations": None,
                    }
                },
                "ignore": [
                    "lm_head",
                    r"re:.*\.self_attn\..*$",
                    r"re:.*\.mlp\.shared_expert\..*$",
                ],
            }
        )

        self.assertNotIn("Linear", quant_config.target_scheme_map)
        scheme = quant_config.get_moe_scheme(
            torch.nn.Module(),
            layer_name="model.layers.1.mlp.experts",
        )

        self.assertIsInstance(
            scheme,
            (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE),
        )
        self.assertEqual(scheme.num_bits, 4)
        self.assertEqual(scheme.group_size, 32)


if __name__ == "__main__":
    unittest.main()
