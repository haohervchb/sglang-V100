// SPDX-License-Identifier: Apache-2.0
// Long-context grouped decode for Volta (SM70), D256 G6 (H6/Hkv1), E5M2 KV.
//
// Read-once split-KV design: grid = (kv_heads, splits, batch). Each CTA owns
// one kv head and one context partition, streaming that partition's K/V once
// with 16-byte vectorized loads from the dense token-major layout. This keeps
// DRAM traffic at the theoretical minimum (unlike per-query-head partition
// schemes that re-read the same KV once per GQA head). A split-merge kernel
// (the existing TileLang combine) rescales the FP32 softmax states, with the
// same partial-output ABI. Numerics mirror _kernels_paged_decode.py: scores
// are scaled by softmax_scale*k_scale, probabilities are exp2((s-m)*scale*log2e),
// and the partial output is normalized by the partition sum and v_scale.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace sm70_longctx {

constexpr int kThreads = 512;
constexpr int kBlockN = 32;          // tokens per tile (smem-bounded)
constexpr int kDim = 256;
constexpr int kPairsPerLane = (kDim / 2) / 32;  // half2 pairs per lane (4)
constexpr int kAccPerLane = 2 * kPairsPerLane;  // fp32 accumulators per lane
constexpr int kGroup = 6;            // heads / kv_heads (TP4 GQA layout)
constexpr int kPageSize = 16;        // KV cache tokens per page
constexpr int kKVStride = 264;       // padded smem row stride (half units)
constexpr int kSmemKV = kBlockN * kKVStride;  // halves per K or V tile
constexpr float kLog2E = 1.4426950408889634f;

// smem (bytes): ks + vs (fp16 tiles) + qs + scores (fp32) + probs (fp16)
constexpr int kSmemBytes = 2 * kSmemKV * sizeof(__half) +
                           kGroup * kDim * sizeof(__half) +
                           kGroup * kBlockN * sizeof(float) +
                           kGroup * kBlockN * sizeof(__half);

__device__ __forceinline__ void e5m2_to_fp16_16(const uint4 raw, half* dst) {
  // E5M2 shares sign/exponent bit positions with FP16: each byte shifted left
  // by 8 is an exact value conversion, and the 16 output halves must keep the
  // same dim order as the 16 input bytes. Each 32-bit input word b = [b0..b3]
  // produces one uint32 pair [fp16(b0), fp16(b1)] and one [fp16(b2), fp16(b3)];
  // the pairs go to consecutive uint32 slots so dim order is preserved.
  const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
  uint32_t out[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t b = r[w];
    out[2 * w] = (b & 0x000000ffu) << 8 | ((b >> 8) & 0x000000ffu) << 24;
    out[2 * w + 1] = ((b >> 16) & 0x000000ffu) << 8 |
                     ((b >> 24) & 0x000000ffu) << 24;
  }
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(out);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(out + 4);
}

__global__ void __launch_bounds__(kThreads, 1)
decode_partial_kernel(const __half* __restrict__ q,
                      const uint8_t* __restrict__ k_cache,
                      const uint8_t* __restrict__ v_cache,
                      const int* __restrict__ page_table,
                      const int* __restrict__ seq_lens, const int max_splits,
                      const int min_tokens_per_split, const int max_blocks,
                      const float score_scale, const float kv_scale,
                      __half* __restrict__ partial_o,
                      float* __restrict__ partial_lse) {
  const int kv_head = blockIdx.x;   // 0 (dense D256 G6 path)
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;

  const int context = seq_lens[seq_id];
  const int active_splits =
      min(max_splits, max(1, (context + min_tokens_per_split - 1) /
                                 min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  const int split_len = (context + active_splits - 1) / active_splits;
  const int split_begin = split_id * split_len;
  const int split_end = min(context, split_begin + split_len);
  if (split_begin >= split_end) {
    return;
  }
  const float scale_log2 = score_scale * kLog2E;

  // Load the six query rows for this kv head into shared memory.
  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    const int r = i / kDim;
    const int d = i % kDim;
    qs[r * kDim + d] = q[(kv_head * kGroup + r) * kDim + d];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];  // fp32 accumulators per lane (dim pair)
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);

    // ---- Cooperative vectorized load + E5M2 -> FP16 convert (K and V) ----
    // kBlockN tokens * 256 bytes = 8KB/tensor. Each thread: 16B (one uint4)
    // per tensor. slot = tid + it*512; byte_off = slot*16; token = byte>>8;
    // d_off = byte & 255. Consecutive threads read consecutive 16B chunks of
    // a 256B token row -> fully coalesced 128B transactions.
#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int slot = tid + it * kThreads;   // 16-byte units [0, kBlockN*16)
      const int byte_off = slot << 4;
      const int tok_local = byte_off >> 8;
      const int d_off = byte_off & 255;
      const int token = tile_begin + tok_local;
      const bool valid = (token < split_end) && (tok_local < kBlockN);
      uint4 raw_k = make_uint4(0, 0, 0, 0);
      uint4 raw_v = make_uint4(0, 0, 0, 0);
      if (valid) {
        const int64_t base =
            page_table != nullptr
                ? (int64_t)page_table[seq_id * max_blocks + (token >> 4)] *
                          (kPageSize * kDim) +
                      (int64_t)(token & (kPageSize - 1)) * kDim + d_off
                : (int64_t)token * kDim + d_off;
        raw_k = *reinterpret_cast<const uint4*>(k_cache + base);
        raw_v = *reinterpret_cast<const uint4*>(v_cache + base);
      }
      const int dst = tok_local * kKVStride + d_off;
      e5m2_to_fp16_16(raw_k, ks + dst);
      e5m2_to_fp16_16(raw_v, vs + dst);
    }
    __syncthreads();

    // ---- QK: each compute warp owns one query row ----
    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      {
        const int n = lane;  // kBlockN == 32 == warp width
        const __half2* krow =
            reinterpret_cast<const __half2*>(ks + n * kKVStride);
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
        for (int h = 0; h < kDim / 8; ++h) {
          const __half2 qh0 = qr[4 * h];
          const __half2 kh0 = krow[4 * h];
          const __half2 qh1 = qr[4 * h + 1];
          const __half2 kh1 = krow[4 * h + 1];
          const __half2 qh2 = qr[4 * h + 2];
          const __half2 kh2 = krow[4 * h + 2];
          const __half2 qh3 = qr[4 * h + 3];
          const __half2 kh3 = krow[4 * h + 3];
          acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
          acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
          acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
          acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
          acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
          acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
          acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
          acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
        }
        const float acc = acc0 + acc1 + acc2 + acc3;
        if (n < tile_tokens) {
          scores[r * kBlockN + n] = acc;
        }
      }
      __syncwarp();
      float loc_max = -1.0e30f;
      if (lane < tile_tokens) {
        loc_max = scores[r * kBlockN + lane];
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float loc_sum = 0.f;
      {
        const int n = lane;
        float p = 0.f;
        if (n < tile_tokens) {
          p = exp2f((scores[r * kBlockN + n] - m_new) * scale_log2);
          probs[r * kBlockN + n] = __float2half(p);
        }
        loc_sum = p;
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    // ---- PV: o_acc += p * V ----
    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 v = vrow[lane + j * 32];
          o_acc[r][2 * j] = fmaf(p, __half2float(v.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(v.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  // ---- Finalize partial output for this partition ----
  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    __half* o_row = partial_o +
                    (((int64_t)seq_id * max_splits + split_id) * kGroup + r) *
                        kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      o_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[r][2 * j] * inv_l * kv_scale);
      o_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l * kv_scale);
    }
  }
  if (is_compute_warp && lane == 0) {
    const int r = warp;
    const float lse =
        l_row[r] > 0.f ? __log2f(l_row[r]) + m_row[r] * scale_log2 : -1.0e30f;
    partial_lse[((int64_t)seq_id * max_splits + split_id) * kGroup + r] = lse;
  }
}

}  // namespace sm70_longctx

void sm70_longctx_decode(torch::Tensor q, torch::Tensor k_cache,
                         torch::Tensor v_cache, torch::Tensor page_table,
                         torch::Tensor seq_lens, int64_t max_splits,
                         int64_t min_tokens_per_split, double softmax_scale,
                         double k_scale, double v_scale,
                         torch::Tensor partial_o,
                         torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  const int heads = q.size(1);
  const int heads_kv = k_cache.size(2);
  const int dim = q.size(2);
  const int batch = q.size(0);
  TORCH_CHECK(heads_kv == 1 && heads == kGroup && dim == kDim,
              "sm70_longctx expects H6/Hkv1/D256 (TP4 GQA) layout");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8 &&
                  v_cache.scalar_type() == torch::kUInt8,
              "sm70_longctx expects E5M2 byte KV cache");
  TORCH_CHECK(q.dtype() == torch::kHalf, "sm70_longctx expects fp16 queries");
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)heads_kv, (unsigned)max_splits, (unsigned)batch);
  const int* page_ptr = page_table.numel() > 0 ? page_table.data_ptr<int>()
                                               : nullptr;
  const int max_blocks = page_table.numel() > 0 ? (int)page_table.size(1) : 0;
  decode_partial_kernel<<<grid, sm70_longctx::kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_ptr,
      seq_lens.data_ptr<int>(), (int)max_splits, (int)min_tokens_per_split,
      max_blocks, (float)(softmax_scale * k_scale), (float)v_scale,
      reinterpret_cast<__half*>(partial_o.data_ptr()),
      partial_lse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm70_longctx_decode", &sm70_longctx_decode,
        "Long-context grouped decode partial (D256 G6 E5M2)");
}
