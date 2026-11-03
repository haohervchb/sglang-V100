# Qwen3.8 Next speculative decoding experiments

This harness uses one saved token corpus for cold-cache comparisons at 1,000,
8,192, 25,000 and 70,000 input tokens. It includes ordinary prose and a synthetic
copy-heavy code edit. The code case is useful for prompt lookup; it is not an
OpenCode task suite or evidence of a general coding speedup.

Run from the repository root in the `sglang-v100` environment:

```bash
python benchmark/qwen38_spec/bench.py corpus --output /tmp/qwen-corpus.json
python benchmark/qwen38_spec/bench.py run \
  --corpus /tmp/qwen-corpus.json --label mtp \
  --output /tmp/mtp.jsonl --repeats 2
```

The server URL defaults to `http://127.0.0.1:8082`. Each request waits for an
acknowledged cache flush, then checks exact input/output token counts, cache
hits and retractions. Warmups are saved and excluded from comparisons. The
default output length is 512 tokens; both tasks are designed to continue beyond
that length. Longer forced outputs can run beyond the natural end of the code
edit and must not be presented as useful code throughput.

Records contain the input token hash, request ID, TTFT, end-to-end throughput,
decode throughput, streamed token counts, native acceptance histograms and
output text. Decode rate excludes the first streamed burst. Cycle time and
tokens per cycle use verification-counter differences over that same interval;
they are client observations, not isolated kernel timings. Acceptance length
includes the bonus token. Server configuration is saved next to each JSONL file.

## Small SPEED-Bench screen

Generate a pinned 20-case first-turn subset covering coding, reasoning,
writing, retrieval and multilingual prompts:

```bash
python benchmark/qwen38_spec/speed_corpus.py \
  --output /tmp/speed-corpus.json --manifest /tmp/speed-manifest.json
python benchmark/qwen38_spec/bench.py run \
  --corpus /tmp/speed-corpus.json --label mtp-speed \
  --output /tmp/mtp-speed.jsonl --natural-eos --repeats 2 --warmups 0
```

This is a screening subset, not a full SPEED-Bench score or agent-session
replay. `--natural-eos` permits normal completion and records actual output
length; a response delivered in a single burst has no decode-rate estimate.
The manifest records dataset revision, question IDs, input hashes and lengths.
Keep the downloaded dataset and generated prompt corpus local.

## Experimental switches

Both switches default to **off**:

- `SGLANG_SM70_QSA_FUSED_GRAPH_METADATA=1` adapts upstream
  [#41173](https://github.com/sgl-project/sglang/pull/41173): fuse row/layout
  metadata and prepare compatible MTP draft steps together. Variable draft
  extension keeps its layout pass, and incompatible layouts retain the old path.
- `SGLANG_SM70_PROMPT_LOOKUP=1` enables the single-request, greedy Qwen MTP-3/4
  experiment. An exact eight-token prompt suffix proposes three tokens, within
  the existing four-token verification width. A match skips the MTP draft loop;
  target verification, recurrent-state commit and draft cache extension remain
  unchanged. Verified GPU outputs update CPU lookup history through an async
  copy whose event and sources remain alive until consumed. This coordination
  has a cost even when there is no match, so measure prose as well as code.

Prompt lookup requires SM70, the built-in Qwen MTP model, top-k one,
`--max-running-requests 1`, and disabled adaptive speculation. Sampled,
grammar-constrained and multimodal requests use ordinary MTP. This experiment
does not establish wider-batch, stochastic lookup or adaptive-width support.

On a server launched with prompt lookup enabled, run:

```bash
python benchmark/qwen38_spec/bench.py run \
  --corpus /tmp/qwen-corpus.json --label lookup-ab \
  --output /tmp/lookup-ab.jsonl --repeats 2 --lookup-ab
```

This alternates `sampling_params.custom_params.prompt_lookup=false/true` for
each identical prompt and reverses the order in the next repeat. The request
field can only opt out of the server experiment. It does not enable it on an
ordinary server. For a QSA comparison, restart with its environment flag changed;
the target graph captures that choice at startup.

`bench.py sweep` automates process-owned server restarts using a JSON file:

```json
{
  "order": ["a", "b", "b", "a"],
  "variants": {
    "a": {"command": ["bash", "your_launch_script.sh"], "env": {"SGLANG_SM70_QSA_FUSED_GRAPH_METADATA": "0"}},
    "b": {"command": ["bash", "your_launch_script.sh"], "env": {"SGLANG_SM70_QSA_FUSED_GRAPH_METADATA": "1"}}
  }
}
```

Use `--config`, `--corpus`, and `--output-dir`. Commands are argument arrays,
never shell-evaluated strings. The harness refuses to start over a listening
server and only terminates process groups it launched. Use no simulated
acceptance (`SGLANG_SIMULATE_ACC_LEN=-1`), no competing GPU workload, and the
same model revision, launch configuration and output length for every variant.

An isolated worktree does not contain ignored compiled extensions. Set
`SGLANG_SM70_TURBOMIND_OPS_PATH` to the same built library in every variant,
and verify the Marlin libraries too. A variant can require boot-log markers
before measuring, for example
`"required_log_markers": ["SM70 FP16 MoE: using TurboMind batched GEMM"]`.
Missing markers abort the sweep instead of benchmarking a silent fallback.
Pin both `--revision` and `--speculative-draft-model-revision`.

## Resumable V100 round

`round.py` runs a JSON-configured sequence of process-owned servers, checks
required backend markers, and archives each completed stage immediately. Put
the output directory on persistent storage; `/tmp` can disappear after reboot.
The config contains `corpus`, `speed_corpus`, and a `variants` list with `name`,
`cwd`, `command` (an argument array), `env`, and `required_log_markers`.
Optional `cases`, `repeats`, `speed`, `multimodal` and `profile` fields bound a
screening run. Profile activity occurs only after throughput and serving gates.

```bash
python benchmark/qwen38_spec/round.py \
  --config /persistent/qwen-round/matrix.json \
  --output-dir /persistent/qwen-round \
  --archive-dir benchmark/my-qwen-round/raw
python benchmark/qwen38_spec/summarize.py /persistent/qwen-round \
  --output /persistent/qwen-round/summary.json
python benchmark/qwen38_spec/compare.py /persistent/qwen-round \
  --baseline e5m2_4 --output /persistent/qwen-round/paired.json
```

`compare.py` checks matching input hashes, temperatures and output limits,
then compares each prompt's median decode rate. Its domain summaries are
geometric means of those paired ratios. SPEED response text stays local;
published records retain its hash, token counts and timing events.

## Separate profiling

For diagnostic runs, prepend `benchmark/qwen38_spec/probe` to `PYTHONPATH` and
set `SGLANG_SPEC_PROBE_DIR` to an absolute output directory. The probe wraps the
worker only in that process; it does not install a production runtime hook.
It attaches after the worker's normal import, so it does not import Torch or
freeze backend choices before server setup.
It records rank-zero CUDA events for draft, verify, target forward, recurrent
state commit, draft extend and the complete round, plus CPU submission time.
Events are queried in later rounds without a profiling synchronization per round.
`verify` includes `target` and `state_commit`; these nested timings must not be
added together. `SGLANG_SPEC_PROBE_ROUNDS` caps the records (default 512).

With `SGLANG_SPEC_PROBE_ON_PROFILE=1`, event timing is active only while the
server's PyTorch profiler is active. Otherwise every eligible round is timed.
Use `/start_profile` and `/stop_profile` to obtain a Chrome trace, with
`activities=["CPU", "GPU"]`, `with_stack=false` and `record_shapes=false`.
Use `collect_profile.py --cpu-only` for CUDA-event phase timing without GPU
kernel tracing; this reduces profiler overhead but remains diagnostic.
Issue one more short request after stopping to drain completed event records.
Mark profiled client runs with `--diagnostic`; do not mix their throughput into
the unprofiled A/B result.

`collect_profile.py` performs the warmup, profile start/stop and final drain request:

```bash
python benchmark/qwen38_spec/collect_profile.py --corpus /tmp/qwen-corpus.json \
  --case code_edit_8192 --lookup on --label code-lookup \
  --output-dir /tmp/probe
```

```bash
python benchmark/qwen38_spec/analyze_profile.py \
  /tmp/probe/cycles_tp0_123.jsonl /tmp/probe/trace.json.gz \
  --output /tmp/profile-summary.json
```

GPU idle time is the complement of the union of kernel/memory intervals on
each recorded device, between first and last GPU activity. It excludes CPU API
durations and handles overlapping streams. Kernel-name categories separate
QSA metadata, QSA compute, GDN, MoE, all-reduce and copies; the largest kernel
names are retained for auditing. A trace including prefill describes that whole
profile window. If phase annotations and CUDA correlations are present, a
separate `decode_window` follows the GPU operations submitted during those
phases through their async completion. It excludes prefill and later requests,
while counting all GPU activity inside that window. CUDA event round times
alone cannot establish GPU idle time. All profiler results remain diagnostic
because tracing itself changes launch costs.
Phase summaries are also grouped by request ID so separate workloads and
lookup modes are not averaged together.

## Correctness gates

```bash
python benchmark/qwen38_spec/check.py --output /tmp/checks.json --full-context
python benchmark/qwen38_nvfp4_v100_multimodal_20260909/validate.py \
  --tag candidate --output-dir /tmp/mm-checks
```

The first command checks short answers, grammar and sampled request completion,
retrieval from 262,000 tokens, and a subsequent short request without flushing.
These are smoke tests, not a distributional quality evaluation. Pair them with
the real chain-verifier rejection tests and the graph metadata equivalence tests.
Record baseline failures as well as candidate failures; do not silently loosen
the expected answers to make a gate pass.

For long-document serving, also exercise queued clients and retained caches:

```bash
python benchmark/qwen38_spec/check_queue.py \
  --output-dir /persistent/qwen-checks/queue
```

This sends four clients with six requests each, growing document inputs from
roughly 19K–95K to 34K–110K tokens. It checks exact archive-key retrieval,
records prefix-cache hits and request latency, and samples free GPU memory via
NVML every 100 ms. It does not flush caches between requests. The server may
still process one request at a time; this tests its queue and repeated prefill
allocations, not four-way model execution. These synthetic text requests do not
validate PDF extraction or vision memory use for documents sent as images.
