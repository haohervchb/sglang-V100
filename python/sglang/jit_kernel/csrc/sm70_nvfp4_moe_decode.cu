// SPDX-License-Identifier: Apache-2.0
// Small-batch NVFP4 MoE decode for Qwen3.8 Flash Next on Volta (SM70).
//
// Marlin is an excellent general grouped-GEMM kernel, but this model's TP4
// decode shape is unusually skinny: ten independently routed rows, K=2560,
// and a local expert width of only 160.  Padding every selected expert to an
// 8-row MMA tile leaves most tensor-core work empty.  These kernels instead
// stream the already-repacked Marlin weights as logical groups of eight output
// columns.  Split-K exposes enough parallelism for gate/up; down runs one
// independent work item per (route, 8 output columns).  The dot products use
// native half2 FMA; FP32 reductions fuse SwiGLU and route-weighted summation.
//
// decode() runs two launches (see the vec:: kernels): gate/up with the forty
// split partials folded in shared memory, then down with SwiGLU, the fifty
// route/split partials folded in shared memory, and FP16 output. Results are
// bit-identical to the original four launches, which remain as
// decode_legacy() and as the fallback for operands that are not 16-byte
// aligned.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>

namespace sm70_nvfp4_moe {

constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kHidden = 2560;
constexpr int kIntermediate = 160;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kGroupSize = 16;
constexpr int kSplitK = 40;
constexpr int kDownSplitK = 5;
// At batch one, 256 threads give gate/up only 63 CTAs for 80 Volta SMs.
// Two warps per CTA expose 250 CTAs without changing the reduction order.
constexpr int kThreads = 64;
constexpr float kMarlinScaleCompensation = 1.0f;

__device__ __forceinline__ void warp_argmax(float& value, int& index) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_value > value ||
        (other_value == value && other_index < index)) {
      value = other_value;
      index = other_index;
    }
  }
}

template <typename T>
__device__ __forceinline__ float load_logit(const T* logits, int index) {
  return static_cast<float>(logits[index]);
}

template <>
__device__ __forceinline__ float load_logit<__half>(const __half* logits,
                                                     int index) {
  return __half2float(logits[index]);
}

template <typename T>
__global__ __launch_bounds__(256, 1)
void topk10_softmax_kernel(const T* __restrict__ logits,
                           float* __restrict__ topk_weights,
                           int* __restrict__ topk_ids) {
  // Eight warps independently retain the best ten of 64 logits. Warp zero
  // then retains the best ten of those 80 candidates. Softmax over the final
  // ten is exactly softmax(all logits) followed by top-k renormalization.
  __shared__ float candidate_values[80];
  __shared__ int candidate_ids[80];
  __shared__ float selected_values[kTopK];
  const int row = static_cast<int>(blockIdx.x);
  logits += row * kExperts;
  topk_weights += row * kTopK;
  topk_ids += row * kTopK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int first_id = warp * 64 + lane;
  const int second_id = first_id + 32;
  float first = load_logit(logits, first_id);
  float second = load_logit(logits, second_id);

#pragma unroll
  for (int pick = 0; pick < kTopK; ++pick) {
    float best = first;
    int best_id = first_id;
    if (second > best || (second == best && second_id < best_id)) {
      best = second;
      best_id = second_id;
    }
    warp_argmax(best, best_id);
    const int winner = __shfl_sync(0xffffffffu, best_id, 0);
    if (lane == 0) {
      candidate_values[warp * kTopK + pick] = best;
      candidate_ids[warp * kTopK + pick] = best_id;
    }
    if (first_id == winner) {
      first = -FLT_MAX;
    }
    if (second_id == winner) {
      second = -FLT_MAX;
    }
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }
  float values[3];
  int ids[3];
#pragma unroll
  for (int item = 0; item < 3; ++item) {
    const int candidate = lane + item * 32;
    values[item] =
        candidate < 80 ? candidate_values[candidate] : -FLT_MAX;
    ids[item] = candidate < 80 ? candidate_ids[candidate] : INT_MAX;
  }
#pragma unroll
  for (int pick = 0; pick < kTopK; ++pick) {
    float best = values[0];
    int best_id = ids[0];
#pragma unroll
    for (int item = 1; item < 3; ++item) {
      if (values[item] > best ||
          (values[item] == best && ids[item] < best_id)) {
        best = values[item];
        best_id = ids[item];
      }
    }
    warp_argmax(best, best_id);
    const int winner = __shfl_sync(0xffffffffu, best_id, 0);
    if (lane == 0) {
      selected_values[pick] = best;
      topk_ids[pick] = best_id;
    }
#pragma unroll
    for (int item = 0; item < 3; ++item) {
      if (ids[item] == winner) {
        values[item] = -FLT_MAX;
      }
    }
  }
  __syncwarp();
  float weight = lane < kTopK
                     ? __expf(selected_values[lane] - selected_values[0])
                     : 0.0f;
  float sum = weight;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  sum = __shfl_sync(0xffffffffu, sum, 0);
  if (lane < kTopK) {
    topk_weights[lane] = weight / sum;
  }
}

__global__ void topk10_radix_half(const half* __restrict__ logits,
                                  float* __restrict__ weights,
                                  int* __restrict__ ids) {
  // CUB's stable sort preserves lower-ID ties when the initial blocked order
  // is expert-ID order. Carry the ID as a 16-bit value instead of adding nine
  // sort-key bits: three six-bit passes suffice for all FP16 logits.
  using Sort = cub::BlockRadixSort<uint16_t, 128, 512 / 128, uint16_t, 6>;
  __shared__ typename Sort::TempStorage temp;
  logits += blockIdx.x * 512;
  weights += blockIdx.x * 10;
  ids += blockIdx.x * 10;
  uint16_t keys[512 / 128], values[512 / 128];
#pragma unroll
  for (int j = 0; j < 512 / 128; ++j) {
    int id = threadIdx.x * (512 / 128) + j;
    uint32_t b = __half_as_ushort(logits[id]);
    if ((b & 0x7fff) == 0) b = 0;
    uint32_t ordered = (b & 0x8000) ? (~b & 0xffff) : (b ^ 0x8000);
    keys[j] = ordered;
    values[j] = id;
  }
  Sort(temp).SortDescendingBlockedToStriped(keys, values, 0, 16);
  if (threadIdx.x < 32) {
    int id = values[0];
    float v = __half2float(logits[id]);
    float mx = __shfl_sync(0xffffffff, v, 0);
    float w = threadIdx.x < 10 ? __expf(v - mx) : 0.f;
    float sum = w;
#pragma unroll
    for (int s = 16; s > 0; s /= 2) sum += __shfl_down_sync(0xffffffff, sum, s);
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (threadIdx.x < 10) {
      ids[threadIdx.x] = id;
      weights[threadIdx.x] = w / sum;
    }
  }
}

__device__ __constant__ int kScaleLogicalToStored[8] = {0, 2, 1, 3,
                                                        4, 6, 5, 7};

__device__ __forceinline__ void dequant_fp4x8(uint32_t packed,
                                              __half2* values) {
  // Expand even/odd nibbles once, then interleave zero bytes into four half2
  // values with native PRMT instructions. This preserves the original packed
  // column order and Marlin's scaled FP4 representation bit for bit.
  const uint32_t even =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  packed >>= 4;
  const uint32_t odd =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  uint32_t result[4] = {
      __byte_perm(even, 0, 0x2404), __byte_perm(odd, 0, 0x2404),
      __byte_perm(even, 0, 0x3414), __byte_perm(odd, 0, 0x3414)};
#pragma unroll
  for (int p = 0; p < 4; ++p)
    values[p] = *reinterpret_cast<__half2*>(&result[p]);
}

// SGLang's SM70 Marlin preprocessing encodes a non-negative scale s as a
// byte whose bits become the FP16 representation of s*128 after << 7.
__device__ __forceinline__ __half2 load_scale_pair(
    const uint8_t* encoded, int logical0, int logical1) {
  return __halves2half2(
      __ushort_as_half(static_cast<uint16_t>(encoded[logical0]) << 7),
      __ushort_as_half(static_cast<uint16_t>(encoded[logical1]) << 7));
}

// Packed scales have eight adjacent bytes per qword. Load the whole group
// once and interleave bytes directly into the original FP16 scale pairs.
// The scalar specialization retains support for unaligned metadata views.
template <bool VectorScales>
__device__ __forceinline__ void load_scales8(const uint8_t* encoded, half2* out) {
  if constexpr (VectorScales) {
    const uint2 raw = *reinterpret_cast<const uint2*>(encoded);
    const uint32_t bits[4] = {
        __byte_perm(raw.x, 0, 0x4240) << 7,
        __byte_perm(raw.x, 0, 0x4341) << 7,
        __byte_perm(raw.y, 0, 0x4240) << 7,
        __byte_perm(raw.y, 0, 0x4341) << 7};
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = *reinterpret_cast<const half2*>(&bits[p]);
  } else {
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = load_scale_pair(encoded, kScaleLogicalToStored[2 * p],
                              kScaleLogicalToStored[2 * p + 1]);
  }
}

template <bool VectorScales>
__global__ void __launch_bounds__(kThreads, 2)
gate_up_partial_kernel(const __half* __restrict__ input,
                       const uint32_t* __restrict__ weight,
                       const uint8_t* __restrict__ scales,
                       const int* __restrict__ topk_ids,
                       int num_routes,
                       float* __restrict__ partials) {
  constexpr int kQwords = kGateUp / 8;
  constexpr int kGroups = kHidden / kGroupSize;
  constexpr int kGroupsPerSplit = kGroups / kSplitK;
  constexpr int kExpertWords = (kHidden / 16) * (kGateUp * 2);

  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = num_routes * kSplitK * kQwords;
  if (work >= kTotal) {
    return;
  }
  const int qword = work % kQwords;
  const int split_route = work / kQwords;
  const int split = split_route % kSplitK;
  const int route = split_route / kSplitK;
  const int token = route / kTopK;
  const int expert = topk_ids[route];
  if (expert < 0 || expert >= kExperts) {
    return;
  }
  const int n_base = qword * 8;
  const uint32_t* expert_weight = weight + expert * kExpertWords;
  const uint8_t* expert_scales =
      scales + static_cast<int64_t>(expert) * kGroups * kGateUp;

  __half2 accum[4] = {
      __float2half2_rn(0.0f), __float2half2_rn(0.0f),
      __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
  const int group_begin = split * kGroupsPerSplit;
#pragma unroll
  for (int group_it = 0; group_it < kGroupsPerSplit; ++group_it) {
    const int group = group_begin + group_it;
    __half2 scale[4];
    load_scales8<VectorScales>(expert_scales + group * kGateUp + n_base, scale);
#pragma unroll
    for (int r = 0; r < kGroupSize; ++r) {
      const int k = group * kGroupSize + r;
      const __half2 x = __halves2half2(
          input[token * kHidden + k], input[token * kHidden + k]);
      const int qword_in_tile = qword & 7;
      const int n_tile = qword >> 3;
      const int offset = group * (kGateUp * 2) + n_tile * 128 +
                         r * 8 + qword_in_tile;
      const uint32_t packed = expert_weight[offset];
      __half2 value[4];
      dequant_fp4x8(packed, value);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        accum[p] = __hfma2(x, __hmul2(scale[p], value[p]), accum[p]);
      }
    }
  }
  float* out = partials +
               ((split * num_routes + route) * kGateUp + n_base);
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    out[2 * p] = __half2float(accum[p].x);
    out[2 * p + 1] = __half2float(accum[p].y);
  }
}

__global__ void __launch_bounds__(kThreads, 2)
gate_up_reduce_silu_kernel(const float* __restrict__ partials,
                           const float* __restrict__ global_scales,
                           const int* __restrict__ topk_ids,
                           int num_routes,
                           __half* __restrict__ activated) {
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = num_routes * kIntermediate;
  if (work >= kTotal) {
    return;
  }
  const int route = work / kIntermediate;
  const int n = work - route * kIntermediate;
  const int expert = topk_ids[route];
  if (expert < 0 || expert >= kExperts) {
    activated[work] = __float2half(0.0f);
    return;
  }
  float gate = 0.0f;
  float up = 0.0f;
#pragma unroll
  for (int split = 0; split < kSplitK; ++split) {
    const float* part = partials + (split * num_routes + route) * kGateUp;
    gate += part[n];
    up += part[n + kIntermediate];
  }
  const float global =
      global_scales[expert] * kMarlinScaleCompensation;
  gate *= global;
  up *= global;
  activated[work] = __float2half_rn((gate / (1.0f + __expf(-gate))) * up);
}

template <bool VectorScales>
__global__ void __launch_bounds__(kThreads, 2)
down_partial_kernel(const __half* __restrict__ activated,
                    const uint32_t* __restrict__ weight,
                    const uint8_t* __restrict__ scales,
                    const float* __restrict__ global_scales,
                    const int* __restrict__ topk_ids,
                    const float* __restrict__ topk_weights,
                    int num_routes,
                    float* __restrict__ partials) {
  constexpr int kQwords = kHidden / 8;
  constexpr int kGroups = kIntermediate / kGroupSize;
  constexpr int kGroupsPerSplit = kGroups / kDownSplitK;
  constexpr int kExpertWords = (kIntermediate / 16) * (kHidden * 2);

  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = num_routes * kDownSplitK * kQwords;
  if (work >= kTotal) {
    return;
  }
  const int qword = work % kQwords;
  const int split_route = work / kQwords;
  const int split = split_route % kDownSplitK;
  const int route = split_route / kDownSplitK;
  const int expert = topk_ids[route];
  const int n_base = qword * 8;
  float* out = partials +
               (route * kDownSplitK + split) * kHidden + n_base;
  if (expert < 0 || expert >= kExperts) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      out[p] = 0.0f;
    }
    return;
  }
  const uint32_t* expert_weight = weight + expert * kExpertWords;
  const uint8_t* expert_scales =
      scales + static_cast<int64_t>(expert) * kGroups * kHidden;
  const __half* route_input = activated + route * kIntermediate;
  __half2 accum[4] = {
      __float2half2_rn(0.0f), __float2half2_rn(0.0f),
      __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
  const int group_begin = split * kGroupsPerSplit;
#pragma unroll
  for (int group_it = 0; group_it < kGroupsPerSplit; ++group_it) {
    const int group = group_begin + group_it;
    __half2 scale[4];
    load_scales8<VectorScales>(expert_scales + group * kHidden + n_base, scale);
#pragma unroll
    for (int r = 0; r < kGroupSize; ++r) {
      const int k = group * kGroupSize + r;
      const __half2 x = __halves2half2(route_input[k], route_input[k]);
      const int group_n_tile = qword >> 5;
      const int subtile = (qword >> 3) & 3;
      const int qword_in_tile = qword & 7;
      const int offset = group * (kHidden * 2) + group_n_tile * 512 +
                         qword_in_tile * 4 + subtile + r * 32;
      const uint32_t packed = expert_weight[offset];
      __half2 value[4];
      dequant_fp4x8(packed, value);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        accum[p] = __hfma2(x, __hmul2(scale[p], value[p]), accum[p]);
      }
    }
  }
  const float multiplier = global_scales[expert] *
                           kMarlinScaleCompensation * topk_weights[route];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    out[2 * p] = __half2float(accum[p].x) * multiplier;
    out[2 * p + 1] = __half2float(accum[p].y) * multiplier;
  }
}

__global__ void __launch_bounds__(kThreads, 2)
down_reduce_kernel(const float* __restrict__ partials,
                   int batch_size,
                   __half* __restrict__ output) {
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (work >= batch_size * kHidden) {
    return;
  }
  const int token = work / kHidden;
  const int n = work - token * kHidden;
  float sum = 0.0f;
#pragma unroll
  for (int token_route = 0; token_route < kTopK; ++token_route) {
    const int route = token * kTopK + token_route;
#pragma unroll
    for (int split = 0; split < kDownSplitK; ++split) {
      sum += partials[(route * kDownSplitK + split) * kHidden + n];
    }
  }
  output[work] = __float2half_rn(sum);
}

// ---------------------------------------------------------------------------
// Two-launch decode path (bit-identical to the four-launch legacy path).
//
// gate_up_fused_kernel: one CTA per (route, 64-column n_tile) owns all forty
// 64-k splits of those columns: 320 threads = 40 splits x 8 qwords. Each
// thread keeps the legacy per-column FP16 HFMA2 chain of its split in
// registers (same products, same k order), loads the 16 x values of a group
// with two 128-bit loads and the 8 scale bytes with one 64-bit load, and parks
// the FP32 split partial in shared memory. Sixty-four threads then fold the
// forty partials in split order (0.0f + p0 + ... + p39), multiply by the
// expert global scale, and store the FP32 pre-activations gate|up of the
// route into gate_up_partials[route][320]. One 4-byte weight word per thread
// per k is deliberate: eight lanes cover each 32-byte (n_tile, r) row, which
// measured faster on V100 DRAM than 8- or 16-byte per-thread loads of the
// same layout (fewer distinct rows in flight per warp).
//
// down_fused_kernel: one CTA per (token, C-column output tile) owns all ten
// routes x five 32-k splits of those columns. Each thread evaluates SwiGLU
// exactly as gate_up_reduce_silu did (FP32 -> FP16), runs the FP16 HFMA2
// chain over QWD packed words per k (uint2/uint4 loads), applies
// global*topk_weight, and parks the FP32 product in shared memory. C threads
// fold the fifty partials in route-major/split-minor order and write FP16
// output. Products are formed with __fmul_rn so they can never contract into
// the fold's adds. No global partials, no reduce launches, no counters.
// ---------------------------------------------------------------------------

namespace vec {

constexpr int kNTiles = kGateUp / 64;                                  // 5
constexpr int kW13ExpertWords = (kHidden / 16) * (kGateUp * 2);        // 102400
constexpr int kW2ExpertWords = (kIntermediate / 16) * (kHidden * 2);   // 51200
constexpr int kGroupsPerSplit1 = (kHidden / kGroupSize) / kSplitK;     // 4
constexpr int kGroupsPerSplit2 = (kIntermediate / kGroupSize) / kDownSplitK;  // 2
constexpr int kRouteSplits = kTopK * kDownSplitK;                      // 50
constexpr int kGateThreads = kSplitK * 8;                              // 320
constexpr int kMaxRoutes = 32 * kTopK;
constexpr int kSmemPad = 4;

template <int N>
__device__ __forceinline__ void load_words(const uint32_t* p, uint32_t (&w)[N]) {
  if constexpr (N == 4) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
  } else if constexpr (N == 2) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(p));
    w[0] = v.x; w[1] = v.y;
  } else {
    w[0] = __ldg(p);
  }
}

// Same byte interleave as load_scales8<true>, from an already-loaded qword.
__device__ __forceinline__ void expand_scales8(uint32_t lo, uint32_t hi,
                                               half2* out) {
  const uint32_t bits[4] = {
      __byte_perm(lo, 0, 0x4240) << 7, __byte_perm(lo, 0, 0x4341) << 7,
      __byte_perm(hi, 0, 0x4240) << 7, __byte_perm(hi, 0, 0x4341) << 7};
#pragma unroll
  for (int p = 0; p < 4; ++p)
    out[p] = *reinterpret_cast<const half2*>(&bits[p]);
}

// Broadcast half k of packed halves into both lanes of a half2: bit-identical
// to __halves2half2(x[k], x[k]).
template <int N>
__device__ __forceinline__ half2 broadcast_half(const uint32_t (&x)[N], int k) {
  const uint32_t b = __byte_perm(x[k >> 1], 0, (k & 1) ? 0x3232 : 0x1010);
  return *reinterpret_cast<const half2*>(&b);
}

// Route at position p of the stable order "valid experts ascending, invalid
// last, ties by route index". Every CTA derives the same bijection from
// topk_ids (deterministic counts + index-ordered ballots), so CTA p picks its
// route without a planning launch or scratch. Routes that share an expert
// land in adjacent CTAs, run concurrently, and share their weight stream
// through L2. Must be called by all T threads of the CTA.
template <int T>
__device__ __forceinline__ int sorted_route(const int* __restrict__ topk_ids,
                                            int n, int p, int* expert_out) {
  constexpr int kPer = (kMaxRoutes + T - 1) / T;  // routes per thread
  constexpr int kBins = kExperts + 1;              // bin 512 = invalid id
  constexpr int kBinsPer = (kBins + T - 1) / T;
  constexpr int kWarps = T / 32;
  __shared__ int s_cnt[kBinsPer * T];
  __shared__ int s_wsum[kWarps];
  __shared__ unsigned s_mask[kPer * kWarps];
  __shared__ int s_sel[2];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int b = 0; b < kBinsPer; ++b) s_cnt[tid * kBinsPer + b] = 0;
  int keys[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int r = tid + i * T;
    keys[i] = -1;
    if (r < n) {
      const int e = topk_ids[r];
      keys[i] = (e >= 0 && e < kExperts) ? e : kExperts;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    if (keys[i] >= 0) atomicAdd(&s_cnt[keys[i]], 1);
  __syncthreads();
  // Exclusive scan of bin counts; the thread owning position p's bin records
  // (bin, rank within bin).
  int c[kBinsPer];
  int v = 0;
#pragma unroll
  for (int b = 0; b < kBinsPer; ++b) {
    c[b] = s_cnt[tid * kBinsPer + b];
    v += c[b];
  }
  const int own = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += y;
  }
  if (lane == 31) s_wsum[warp] = v;
  __syncthreads();
  int start = v - own;
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    if (w < warp) start += s_wsum[w];
#pragma unroll
  for (int b = 0; b < kBinsPer; ++b) {
    if (p >= start && p < start + c[b]) {
      s_sel[0] = tid * kBinsPer + b;
      s_sel[1] = p - start;
    }
    start += c[b];
  }
  __syncthreads();
  const int bin = s_sel[0];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const unsigned m = __ballot_sync(0xffffffffu, keys[i] == bin);
    if (lane == 0) s_mask[i * kWarps + warp] = m;
  }
  __syncthreads();
  // Routes r = i*T + warp*32 + lane enumerate (i, warp, lane) in index order.
  int k = s_sel[1];
  int route = 0;
#pragma unroll 1
  for (int j = 0; j < kPer * kWarps; ++j) {
    unsigned m = s_mask[j];
    const int cnt = __popc(m);
    if (k < cnt) {
      for (int q = 0; q < k; ++q) m &= m - 1;
      const int i = j / kWarps;
      route = i * T + (j - i * kWarps) * 32 + (__ffs(m) - 1);
      break;
    }
    k -= cnt;
  }
  *expert_out = bin;
  return route;
}

// sorted != 0: CTA (p, n_tile) handles the route at sorted position p.
template <int MINB>
__global__ void __launch_bounds__(kGateThreads, MINB)
gate_up_fused_kernel(const __half* __restrict__ input,
                     const uint32_t* __restrict__ weight,
                     const uint8_t* __restrict__ scales,
                     const float* __restrict__ global_scales,
                     const int* __restrict__ topk_ids, int num_routes,
                     float* __restrict__ pre_act, int sorted) {
  constexpr int kStride = 64 + kSmemPad;
  __shared__ __align__(16) float part_s[kSplitK * kStride];
  const int bid = static_cast<int>(blockIdx.x);
  const int route_pos = bid / kNTiles;
  const int n_tile = bid - route_pos * kNTiles;
  int route, expert;
  if (sorted) {
    route = sorted_route<kGateThreads>(topk_ids, num_routes, route_pos, &expert);
  } else {
    route = route_pos;
    expert = topk_ids[route];
  }
  if (expert < 0 || expert >= kExperts) {
    return;  // down_fused_kernel never reads pre-activations of such routes.
  }
  const int token = route / kTopK;
  const int qword = threadIdx.x & 7;  // qword_in_tile
  const int split = threadIdx.x >> 3;
  const int group0 = split * kGroupsPerSplit1;
  const uint32_t* wp = weight + expert * kW13ExpertWords +
                       group0 * (kGateUp * 2) + n_tile * 128 + qword;
  const uint8_t* sp = scales + static_cast<int64_t>(expert) *
                                   (kHidden / kGroupSize) * kGateUp +
                      group0 * kGateUp + (n_tile * 8 + qword) * 8;
  const __half* xp = input + token * kHidden + group0 * kGroupSize;

  half2 acc[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) acc[p] = __float2half2_rn(0.0f);
#pragma unroll
  for (int g = 0; g < kGroupsPerSplit1; ++g) {
    half2 sc[4];
    {
      const uint2 raw = *reinterpret_cast<const uint2*>(sp + g * kGateUp);
      expand_scales8(raw.x, raw.y, sc);
    }
    uint32_t xr[8];
    {
      const uint4 a = __ldg(reinterpret_cast<const uint4*>(xp + g * 16));
      const uint4 b = __ldg(reinterpret_cast<const uint4*>(xp + g * 16 + 8));
      xr[0] = a.x; xr[1] = a.y; xr[2] = a.z; xr[3] = a.w;
      xr[4] = b.x; xr[5] = b.y; xr[6] = b.z; xr[7] = b.w;
    }
#pragma unroll
    for (int r = 0; r < kGroupSize; ++r) {
      const uint32_t w = __ldg(wp + g * (kGateUp * 2) + r * 8);
      const half2 x = broadcast_half(xr, r);
      half2 v[4];
      dequant_fp4x8(w, v);
#pragma unroll
      for (int p = 0; p < 4; ++p)
        acc[p] = __hfma2(x, __hmul2(sc[p], v[p]), acc[p]);
    }
  }
  float* row = part_s + split * kStride + qword * 8;
#pragma unroll
  for (int p = 0; p < 4; p += 2) {
    float4 v;
    v.x = __low2float(acc[p]);
    v.y = __high2float(acc[p]);
    v.z = __low2float(acc[p + 1]);
    v.w = __high2float(acc[p + 1]);
    *reinterpret_cast<float4*>(row + 2 * p) = v;
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kSplitK; ++s) sum += part_s[s * kStride + threadIdx.x];
    const float global = global_scales[expert] * kMarlinScaleCompensation;
    pre_act[route * kGateUp + n_tile * 64 + threadIdx.x] = sum * global;
  }
}

// C = output columns per CTA (64, 128, or 256 = one Marlin 256-column group
// tile), QWD = packed words per thread per k (1, 2 or 4), NSTAGE = shared
// fold stages (50/NSTAGE partials parked at a time), SACT = evaluate the
// token's 10x160 SwiGLU once per CTA into shared memory instead of per
// thread. CTA order is tile-major (all tokens of one tile adjacent) so tokens
// that share experts stream those weights through L2 together.
template <int C, int QWD, int NSTAGE, int MINB, bool SACT>
__global__ void __launch_bounds__(kRouteSplits * (C / 8) / QWD, MINB)
down_fused_kernel(const float* __restrict__ pre_act,
                  const uint32_t* __restrict__ weight,
                  const uint8_t* __restrict__ scales,
                  const float* __restrict__ global_scales,
                  const int* __restrict__ topk_ids,
                  const float* __restrict__ topk_weights, int batch_size,
                  __half* __restrict__ output) {
  constexpr int kThreadsK2 = kRouteSplits * (C / 8) / QWD;
  constexpr int kTilesPerGnt = 256 / C;
  constexpr int kRowWords = C / 8;          // words of this tile per 32-word row
  constexpr int kParts = kRowWords / QWD;   // threads per (route, split)
  constexpr int kStageRs = kRouteSplits / NSTAGE;
  static_assert(kRouteSplits % NSTAGE == 0, "stages must divide 50");
  static_assert(kThreadsK2 >= C, "need C fold threads");
  constexpr int kStride = C + kSmemPad;
  __shared__ __align__(16) float part_s[kStageRs * kStride];
  const int bid = static_cast<int>(blockIdx.x);
  const int token = bid % batch_size;
  const int tile = bid / batch_size;
  const int gnt = tile / kTilesPerGnt;  // 256-column Marlin group tile
  const int sub = tile - gnt * kTilesPerGnt;
  const int part = threadIdx.x % kParts;
  const int rs = threadIdx.x / kParts;  // route-major / split-minor index
  const int token_route = rs / kDownSplitK;
  const int split = rs - token_route * kDownSplitK;
  const int route = token * kTopK + token_route;
  // Words of one (group, gnt, r) row are ordered qword_in_tile*4 + subtile;
  // this thread owns words word0 .. word0+QWD-1 (one qword_in_tile).
  const int word0 = sub * kRowWords + part * QWD;
  const int qit = word0 >> 2;
  const int st0 = word0 & 3;
  const int expert = topk_ids[route];
  const bool valid = expert >= 0 && expert < kExperts;
  const int group0 = split * kGroupsPerSplit2;

  uint32_t act[16];  // this split's 32 FP16 activations, packed in pairs
  if constexpr (SACT) {
    __shared__ __align__(16) __half act_s[kTopK * kIntermediate];
    for (int i = threadIdx.x; i < kTopK * kIntermediate / 2; i += kThreadsK2) {
      const int rl = i / (kIntermediate / 2);
      const int k = (i - rl * (kIntermediate / 2)) * 2;
      const int rt = token * kTopK + rl;
      const int e = topk_ids[rt];
      if (e >= 0 && e < kExperts) {
        const float2 gt =
            *reinterpret_cast<const float2*>(pre_act + rt * kGateUp + k);
        const float2 up = *reinterpret_cast<const float2*>(
            pre_act + rt * kGateUp + kIntermediate + k);
        const __half a0 =
            __float2half_rn((gt.x / (1.0f + __expf(-gt.x))) * up.x);
        const __half a1 =
            __float2half_rn((gt.y / (1.0f + __expf(-gt.y))) * up.y);
        *reinterpret_cast<__half2*>(act_s + rl * kIntermediate + k) =
            __halves2half2(a0, a1);
      }
    }
    __syncthreads();
    if (valid) {
      const uint4* ap = reinterpret_cast<const uint4*>(
          act_s + token_route * kIntermediate + split * 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = ap[i];
        act[4 * i] = v.x; act[4 * i + 1] = v.y;
        act[4 * i + 2] = v.z; act[4 * i + 3] = v.w;
      }
    }
  }

  float vals[QWD * 8];
  if (valid) {
    if constexpr (!SACT) {
      const float* gate_p = pre_act + route * kGateUp + split * 32;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4 gt = *reinterpret_cast<const float4*>(gate_p + 4 * i);
        const float4 up =
            *reinterpret_cast<const float4*>(gate_p + kIntermediate + 4 * i);
        const __half a0 =
            __float2half_rn((gt.x / (1.0f + __expf(-gt.x))) * up.x);
        const __half a1 =
            __float2half_rn((gt.y / (1.0f + __expf(-gt.y))) * up.y);
        const __half a2 =
            __float2half_rn((gt.z / (1.0f + __expf(-gt.z))) * up.z);
        const __half a3 =
            __float2half_rn((gt.w / (1.0f + __expf(-gt.w))) * up.w);
        act[2 * i] = static_cast<uint32_t>(__half_as_ushort(a0)) |
                     (static_cast<uint32_t>(__half_as_ushort(a1)) << 16);
        act[2 * i + 1] = static_cast<uint32_t>(__half_as_ushort(a2)) |
                         (static_cast<uint32_t>(__half_as_ushort(a3)) << 16);
      }
    }
    const uint32_t* wp = weight + expert * kW2ExpertWords +
                         group0 * (kHidden * 2) + gnt * 512 + word0;
    const uint8_t* sp = scales + static_cast<int64_t>(expert) *
                                     (kIntermediate / kGroupSize) * kHidden +
                        group0 * kHidden + (gnt * 32 + st0 * 8 + qit) * 8;
    half2 acc[QWD][4];
#pragma unroll
    for (int j = 0; j < QWD; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) acc[j][p] = __float2half2_rn(0.0f);
#pragma unroll
    for (int g = 0; g < kGroupsPerSplit2; ++g) {
      half2 sc[QWD][4];
#pragma unroll
      for (int j = 0; j < QWD; ++j) {
        // word (qit, st0 + j) is qword gnt*32 + (st0+j)*8 + qit: +64 bytes.
        const uint2 raw =
            *reinterpret_cast<const uint2*>(sp + g * kHidden + j * 64);
        expand_scales8(raw.x, raw.y, sc[j]);
      }
#pragma unroll
      for (int r = 0; r < kGroupSize; ++r) {
        uint32_t w[QWD];
        load_words<QWD>(wp + g * (kHidden * 2) + r * 32, w);
        const half2 x = broadcast_half(act, g * kGroupSize + r);
#pragma unroll
        for (int j = 0; j < QWD; ++j) {
          half2 v[4];
          dequant_fp4x8(w[j], v);
#pragma unroll
          for (int p = 0; p < 4; ++p)
            acc[j][p] = __hfma2(x, __hmul2(sc[j][p], v[p]), acc[j][p]);
        }
      }
    }
    const float multiplier = global_scales[expert] * kMarlinScaleCompensation *
                             topk_weights[route];
#pragma unroll
    for (int j = 0; j < QWD; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        vals[j * 8 + 2 * p] = __fmul_rn(__low2float(acc[j][p]), multiplier);
        vals[j * 8 + 2 * p + 1] =
            __fmul_rn(__high2float(acc[j][p]), multiplier);
      }
  } else {
#pragma unroll
    for (int i = 0; i < QWD * 8; ++i) vals[i] = 0.0f;  // legacy wrote zeros
  }
  // Shared layout per (route, split): (word - sub*kRowWords)*8 + column.
  const int t = threadIdx.x;
  float sum = 0.0f;
#pragma unroll
  for (int stage = 0; stage < NSTAGE; ++stage) {
    if (rs >= stage * kStageRs && rs < (stage + 1) * kStageRs) {
      float* row = part_s + (rs - stage * kStageRs) * kStride + part * QWD * 8;
#pragma unroll
      for (int i = 0; i < QWD * 8; i += 4)
        *reinterpret_cast<float4*>(row + i) =
            make_float4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
    }
    __syncthreads();
    if (t < C) {
#pragma unroll
      for (int i = 0; i < kStageRs; ++i) sum += part_s[i * kStride + t];
    }
    if (stage + 1 < NSTAGE) __syncthreads();
  }
  if (t < C) {
    const int w = sub * kRowWords + (t >> 3);
    const int col = gnt * 256 + (w & 3) * 64 + (w >> 2) * 8 + (t & 7);
    output[token * kHidden + col] = __float2half_rn(sum);
  }
}

}  // namespace vec

static bool aligned_to(const void* p, uintptr_t a) {
  return (reinterpret_cast<uintptr_t>(p) % a) == 0;
}
void decode_legacy(torch::Tensor input, torch::Tensor w13, torch::Tensor w2,
            torch::Tensor w13_scales, torch::Tensor w2_scales,
            torch::Tensor w13_global, torch::Tensor w2_global,
            torch::Tensor topk_ids, torch::Tensor topk_weights,
            torch::Tensor gate_up_partials, torch::Tensor activated,
            torch::Tensor down_partials, torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= 32 && input.size(1) == kHidden,
              "input must be CUDA FP16 [M, 2560], 1 <= M <= 32");
  const int batch_size = static_cast<int>(input.size(0));
  const int num_routes = batch_size * kTopK;
  TORCH_CHECK(w13.is_cuda() && w13.scalar_type() == at::kInt,
              "w13 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w2.is_cuda() && w2.scalar_type() == at::kInt,
              "w2 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w13_scales.is_cuda() && w13_scales.element_size() == 1,
              "w13_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w2_scales.is_cuda() && w2_scales.element_size() == 1,
              "w2_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w13_global.scalar_type() == at::kFloat &&
                  w2_global.scalar_type() == at::kFloat,
              "global scales must be FP32");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == num_routes,
              "topk_ids must be int32 [M, 10]");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.numel() == num_routes,
              "topk_weights must be FP32 [M, 10]");
  TORCH_CHECK(gate_up_partials.scalar_type() == at::kFloat &&
                  gate_up_partials.numel() == kSplitK * num_routes * kGateUp,
              "gate_up_partials has the wrong shape");
  TORCH_CHECK(activated.scalar_type() == at::kHalf &&
                  activated.numel() == num_routes * kIntermediate,
              "activated has the wrong shape");
  TORCH_CHECK(down_partials.scalar_type() == at::kFloat &&
                  down_partials.numel() == num_routes * kDownSplitK * kHidden,
              "down_partials has the wrong shape");
  TORCH_CHECK(output.scalar_type() == at::kHalf &&
                  output.numel() == batch_size * kHidden,
              "output has the wrong shape");

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int gate_up_work = num_routes * kSplitK * (kGateUp / 8);
  const auto gate_kernel = (reinterpret_cast<uintptr_t>(w13_scales.data_ptr()) % 8 == 0)
                               ? gate_up_partial_kernel<true>
                               : gate_up_partial_kernel<false>;
  gate_kernel<<<(gate_up_work + kThreads - 1) / kThreads, kThreads,
                           0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w13.data_ptr<int>()),
      reinterpret_cast<const uint8_t*>(w13_scales.data_ptr()),
      topk_ids.data_ptr<int>(), num_routes, gate_up_partials.data_ptr<float>());
  const int activated_work = num_routes * kIntermediate;
  gate_up_reduce_silu_kernel<<<
      (activated_work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      gate_up_partials.data_ptr<float>(), w13_global.data_ptr<float>(),
      topk_ids.data_ptr<int>(), num_routes,
      reinterpret_cast<__half*>(activated.data_ptr<at::Half>()));
  const int down_work = num_routes * kDownSplitK * (kHidden / 8);
  const auto down_kernel = (reinterpret_cast<uintptr_t>(w2_scales.data_ptr()) % 8 == 0)
                               ? down_partial_kernel<true>
                               : down_partial_kernel<false>;
  down_kernel<<<(down_work + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w2.data_ptr<int>()),
      reinterpret_cast<const uint8_t*>(w2_scales.data_ptr()),
      w2_global.data_ptr<float>(), topk_ids.data_ptr<int>(),
      topk_weights.data_ptr<float>(), num_routes, down_partials.data_ptr<float>());
  const int output_work = batch_size * kHidden;
  down_reduce_kernel<<<(output_work + kThreads - 1) / kThreads, kThreads, 0,
                       stream>>>(down_partials.data_ptr<float>(), batch_size,
                                reinterpret_cast<__half*>(
                                    output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void decode(torch::Tensor input, torch::Tensor w13, torch::Tensor w2,
            torch::Tensor w13_scales, torch::Tensor w2_scales,
            torch::Tensor w13_global, torch::Tensor w2_global,
            torch::Tensor topk_ids, torch::Tensor topk_weights,
            torch::Tensor gate_up_partials, torch::Tensor activated,
            torch::Tensor down_partials, torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= 32 && input.size(1) == kHidden,
              "input must be CUDA FP16 [M, 2560], 1 <= M <= 32");
  const int batch_size = static_cast<int>(input.size(0));
  const int num_routes = batch_size * kTopK;
  const bool legacy_scratch =
      gate_up_partials.numel() == kSplitK * num_routes * kGateUp &&
      activated.numel() == num_routes * kIntermediate &&
      down_partials.numel() == num_routes * kDownSplitK * kHidden;
  const bool vector_ok =
      aligned_to(input.data_ptr(), 16) && aligned_to(w13.data_ptr(), 16) &&
      aligned_to(w2.data_ptr(), 16) && aligned_to(w13_scales.data_ptr(), 16) &&
      aligned_to(w2_scales.data_ptr(), 8) &&
      aligned_to(gate_up_partials.data_ptr(), 16);
  if (!vector_ok) {
    TORCH_CHECK(legacy_scratch,
                "unaligned operands need the legacy scratch buffers");
    decode_legacy(input, w13, w2, w13_scales, w2_scales, w13_global, w2_global,
                  topk_ids, topk_weights, gate_up_partials, activated,
                  down_partials, output);
    return;
  }
  TORCH_CHECK(w13.is_cuda() && w13.scalar_type() == at::kInt,
              "w13 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w2.is_cuda() && w2.scalar_type() == at::kInt,
              "w2 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w13_scales.is_cuda() && w13_scales.element_size() == 1,
              "w13_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w2_scales.is_cuda() && w2_scales.element_size() == 1,
              "w2_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w13_global.scalar_type() == at::kFloat &&
                  w2_global.scalar_type() == at::kFloat,
              "global scales must be FP32");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == num_routes,
              "topk_ids must be int32 [M, 10]");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.numel() == num_routes,
              "topk_weights must be FP32 [M, 10]");
  TORCH_CHECK(gate_up_partials.is_cuda() &&
                  gate_up_partials.scalar_type() == at::kFloat &&
                  gate_up_partials.numel() >= num_routes * kGateUp,
              "gate_up_partials needs at least M*10*320 FP32 elements");
  TORCH_CHECK(output.scalar_type() == at::kHalf &&
                  output.numel() == batch_size * kHidden,
              "output has the wrong shape");

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const auto* x = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
  const auto* w13p = reinterpret_cast<const uint32_t*>(w13.data_ptr<int>());
  const auto* s13p = reinterpret_cast<const uint8_t*>(w13_scales.data_ptr());
  const auto* w2p = reinterpret_cast<const uint32_t*>(w2.data_ptr<int>());
  const auto* s2p = reinterpret_cast<const uint8_t*>(w2_scales.data_ptr());
  const int* ids = topk_ids.data_ptr<int>();
  float* pre = gate_up_partials.data_ptr<float>();
  auto* out = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

  // Gate/up: CTA = (route, 64-column n_tile) x all 40 splits. From five rows
  // on, CTAs visit routes in expert order so duplicate experts stream
  // through L2 together; below that every CTA is resident at once anyway.
  // The in-kernel expert sort keeps one route per thread for at most
  // kMaxRoutes routes; unsorted routes would leave pre-activations unset.
  TORCH_CHECK(num_routes <= vec::kMaxRoutes,
              "fused MoE decode supports at most 32 rows");
  const int k1_sorted = batch_size <= 4 ? 0 : 1;
  vec::gate_up_fused_kernel<3>
      <<<num_routes * vec::kNTiles, vec::kGateThreads, 0, stream>>>(
          x, w13p, s13p, w13_global.data_ptr<float>(), ids, num_routes, pre,
          k1_sorted);
  // Down: CTA = (token, column tile) x all 10 routes x 5 splits. Tile-major
  // order keeps every token's CTA for one tile resident together.
  const float* g2 = w2_global.data_ptr<float>();
  const float* tw = topk_weights.data_ptr<float>();
  // Tile shapes per row count were picked by measurement on V100 (cold
  // 512-expert weights): wider tiles keep each warp's loads inside fewer
  // DRAM rows; narrow tiles keep small batches spread over all 80 SMs.
#define MOE_VEC_LAUNCH_DOWN(C, QWD, NS, MB, SA)                              \
  vec::down_fused_kernel<C, QWD, NS, MB, SA>                                 \
      <<<batch_size * (kHidden / C), vec::kRouteSplits * (C / 8) / QWD, 0,   \
         stream>>>(pre, w2p, s2p, g2, ids, tw, batch_size, out)
  if (batch_size <= 2 || batch_size == 4) {
    MOE_VEC_LAUNCH_DOWN(64, 1, 1, 2, false);   // 40*M CTAs x 400 threads
  } else if (batch_size == 3) {
    MOE_VEC_LAUNCH_DOWN(128, 2, 2, 2, false);  // 60 CTAs x 400 threads
  } else {
    MOE_VEC_LAUNCH_DOWN(256, 2, 2, 1, true);   // 10*M CTAs x 800 threads
  }
#undef MOE_VEC_LAUNCH_DOWN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void topk10_softmax(torch::Tensor logits, torch::Tensor topk_weights,
                    torch::Tensor topk_ids) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2 &&
                  logits.size(0) >= 1 && logits.size(0) <= 32 &&
                  logits.size(1) == kExperts,
              "logits must be contiguous CUDA [M, 512], 1 <= M <= 32");
  const int batch_size = static_cast<int>(logits.size(0));
  TORCH_CHECK(logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kFloat,
              "logits must be FP16 or FP32");
  TORCH_CHECK(topk_weights.is_cuda() &&
                  topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.numel() == batch_size * kTopK,
              "topk_weights must be CUDA FP32 [M, 10]");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == batch_size * kTopK,
              "topk_ids must be CUDA int32 [M, 10]");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(logits.get_device());
  if (logits.scalar_type() == at::kHalf) {
    topk10_radix_half<<<batch_size, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>());
  } else {
    topk10_softmax_kernel<<<batch_size, 256, 0, stream>>>(
        logits.data_ptr<float>(), topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sm70_nvfp4_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &sm70_nvfp4_moe::decode,
        "Qwen3.8 small-batch SM70 NVFP4 MoE decode");
  m.def("decode_legacy", &sm70_nvfp4_moe::decode_legacy,
        "Legacy four-launch decode");
  m.def("topk10_softmax", &sm70_nvfp4_moe::topk10_softmax,
        "Qwen3.8 small-batch SM70 top-k softmax");
}
