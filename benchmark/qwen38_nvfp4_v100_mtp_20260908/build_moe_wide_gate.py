import subprocess
from pathlib import Path

p = Path("/tmp/qwen38_mtp_20260908")
s = (p / "moe_scale_vector_t64.cu").read_text()
a = s.index("__global__ void __launch_bounds__(kThreads, 2)\ngate_up_partial_kernel")
b = s.index(
    "__global__ void __launch_bounds__(kThreads, 2)\ngate_up_reduce_silu_kernel", a
)
body = """__global__ void __launch_bounds__(kThreads, 2)
gate_up_partial_kernel(const __half* __restrict__ input,
                       const uint32_t* __restrict__ weight,
                       const uint8_t* __restrict__ scales,
                       const int* __restrict__ topk_ids,
                       int num_routes,
                       float* __restrict__ partials) {
  constexpr int Q = WIDE_Q;
  constexpr int kQwords = kGateUp / (8 * Q);
  constexpr int kGroups = kHidden / kGroupSize;
  constexpr int kGroupsPerSplit = kGroups / kSplitK;
  constexpr int kExpertWords = (kHidden / 16) * (kGateUp * 2);
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = num_routes * kSplitK * kQwords;
  if (work >= kTotal) return;
  const int qword = (work % kQwords) * Q;
  const int split_route = work / kQwords;
  const int split = split_route % kSplitK;
  const int route = split_route / kSplitK;
  const int token = route / kTopK;
  const int expert = topk_ids[route];
  if (expert < 0 || expert >= kExperts) return;
  const int n_base = qword * 8;
  const uint32_t* expert_weight = weight + expert * kExpertWords;
  const uint8_t* expert_scales = scales + static_cast<int64_t>(expert) * kGroups * kGateUp;
  __half2 accum[Q][4];
#pragma unroll
  for(int q=0;q<Q;++q)
#pragma unroll
    for(int p=0;p<4;++p) accum[q][p]=__float2half2_rn(0.0f);
  const int group_begin = split * kGroupsPerSplit;
#pragma unroll
  for (int group_it = 0; group_it < kGroupsPerSplit; ++group_it) {
    const int group = group_begin + group_it;
    __half2 scale[Q][4];
#pragma unroll
    for(int q=0;q<Q;++q) load_scale8(expert_scales + group * kGateUp + n_base + q*8, scale[q]);
#pragma unroll
    for (int r = 0; r < kGroupSize; ++r) {
      const int k = group * kGroupSize + r;
      const __half2 x = __halves2half2(input[token * kHidden + k], input[token * kHidden + k]);
      const int qword_in_tile = qword & 7;
      const int n_tile = qword >> 3;
      const int offset = group * (kGateUp * 2) + n_tile * 128 + r * 8 + qword_in_tile;
      const PACKED_TYPE packed = *reinterpret_cast<const PACKED_TYPE*>(expert_weight + offset);
      const uint32_t* words = reinterpret_cast<const uint32_t*>(&packed);
#pragma unroll
      for(int q=0;q<Q;++q) {
        __half2 value[4];
        dequant_fp4x8(words[q], value);
#pragma unroll
        for (int p = 0; p < 4; ++p)
          accum[q][p] = __hfma2(x, __hmul2(scale[q][p], value[p]), accum[q][p]);
      }
    }
  }
  float* out = partials + ((split * num_routes + route) * kGateUp + n_base);
#pragma unroll
  for(int q=0;q<Q;++q)
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      out[q*8+2*p] = __half2float(accum[q][p].x);
      out[q*8+2*p+1] = __half2float(accum[q][p].y);
    }
}

"""
for q in (2, 4):
    for nt in (32, 64, 128):
        name = f"moe_wide_gate_q{q}_t{nt}"
        code = (
            s[:a]
            + body.replace("WIDE_Q", str(q)).replace("PACKED_TYPE", f"uint{q}")
            + s[b:]
        ).replace("kThreads = 64", f"kThreads = {nt}")
        (p / f"{name}.cu").write_text(code)
        with (p / f"{name}.build.log").open("w") as log:
            subprocess.run(
                [
                    "/usr/local/cuda-12.8/bin/nvcc",
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-arch=sm_70",
                    "-Xptxas=-v",
                    "-cubin",
                    str(p / f"{name}.cu"),
                    "-o",
                    str(p / f"{name}.cubin"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("compiled", name, flush=True)
(p / "moe_wide_gate_build.done").write_text("done")
