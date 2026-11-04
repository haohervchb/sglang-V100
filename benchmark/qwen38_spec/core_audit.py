"""Audit a dedicated server with separate timing, DCGM and traced passes.

The caller owns server lifecycle. Do not run this against an occupied serving
endpoint: benchmark requests flush caches. DCGM is paused only while tracing,
and resumed in finally; no GPU clocks or system permissions are changed.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import pynvml
import requests
from bench import measure


class Monitor:
    def __init__(self, path, fields=None):
        self.path = path
        self.fields = fields
        self.stop = threading.Event()
        self.process = None
        self.thread = None

    def __enter__(self):
        if self.fields:
            self.process = subprocess.Popen(
                [
                    "stdbuf",
                    "-oL",
                    "dcgmi",
                    "dmon",
                    "-i",
                    "0,1,2,3",
                    "-e",
                    ",".join(map(str, self.fields)),
                    "-d",
                    "1000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            def collect():
                with self.path.open("x") as output:
                    for line in self.process.stdout:
                        output.write(
                            json.dumps(
                                dict(
                                    unix_s=time.time(),
                                    fields=self.fields,
                                    raw=line.rstrip(),
                                )
                            )
                            + "\n"
                        )
                        output.flush()
        else:
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]

            def collect():
                with self.path.open("x") as output:
                    while not self.stop.is_set():
                        rows = []
                        for handle in handles:
                            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            rows.append(
                                dict(
                                    free_bytes=mem.free,
                                    gpu_percent=util.gpu,
                                    memory_percent=util.memory,
                                    sm_mhz=pynvml.nvmlDeviceGetClockInfo(
                                        handle, pynvml.NVML_CLOCK_SM
                                    ),
                                    memory_mhz=pynvml.nvmlDeviceGetClockInfo(
                                        handle, pynvml.NVML_CLOCK_MEM
                                    ),
                                    power_w=pynvml.nvmlDeviceGetPowerUsage(handle)
                                    / 1000,
                                    throttle_reasons=pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                                        handle
                                    ),
                                )
                            )
                        output.write(
                            json.dumps(dict(unix_s=time.time(), devices=rows)) + "\n"
                        )
                        output.flush()
                        self.stop.wait(0.2)

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=10)
        if not self.fields:
            pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = {c["id"]: c for c in json.loads(args.corpus.read_text())["cases"]}
    selected = [
        "prose_1000",
        "prose_8192",
        "prose_25000",
        "prose_70000",
        "prose_160000",
        "code_edit_8192",
    ]
    session = requests.Session()
    session.trust_env = False
    start = time.monotonic()
    while True:
        try:
            if session.get(args.url + "/health", timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() - start > 1500:
            raise TimeoutError("Audit server readiness timed out")
        time.sleep(5)
    info = session.get(args.url + "/get_server_info", timeout=30)
    info.raise_for_status()
    (args.output_dir / "server_info.json").write_text(
        json.dumps(info.json(), indent=2) + "\n"
    )

    def run_case(name, label, tokens=1024):
        row = measure(args.url, cases[name], tokens)
        row.update(label=label, diagnostic=label.startswith("trace"))
        with (args.output_dir / "requests.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "case_id",
                        "label",
                        "ttft_s",
                        "decode_tok_s",
                        "observed_cycle_ms",
                        "observed_tokens_per_cycle",
                    ]
                }
            ),
            flush=True,
        )
        return row

    with Monitor(args.output_dir / "nvml.jsonl"):
        # Warm all shapes before comparing hardware-counter passes.
        for name in selected:
            run_case(name, "warmup", 32)
        for repeat in range(2):
            for name in ["prose_1000", "prose_8192", "prose_70000", "code_edit_8192"]:
                run_case(name, f"baseline_{repeat}", 512)

        for label, fields in [
            ("dcgm_sm", [1001, 1002, 1003, 1005, 1009, 1010, 1011, 1012]),
            ("dcgm_tensor", [1001, 1004, 1005, 1009, 1010, 1011, 1012]),
        ]:
            with Monitor(args.output_dir / (label + ".jsonl"), fields):
                time.sleep(3)
                for name in selected:
                    run_case(name, label)
                time.sleep(2)

        # DCGM profiling and CUPTI tracing contend for hardware resources.
        # Restore the host-wide profiling state even if a request/export fails.
        subprocess.run(["dcgmi", "profile", "--pause"], check=True)
        try:
            for activities, label in [
                (["CPU"], "trace_cpu"),
                (["CPU", "GPU"], "trace_gpu"),
            ]:
                for name in ["prose_8192", "prose_70000"]:
                    trace_dir = args.output_dir / (label + "_" + name)
                    trace_dir.mkdir()
                    response = session.post(
                        args.url + "/start_profile",
                        json=dict(
                            output_dir=str(trace_dir.resolve()),
                            activities=activities,
                            with_stack=False,
                            record_shapes=False,
                            profile_prefix=label + "_" + name,
                        ),
                        timeout=120,
                    )
                    response.raise_for_status()
                    try:
                        run_case(name, label, 128)
                    finally:
                        session.post(
                            args.url + "/stop_profile", timeout=600
                        ).raise_for_status()
                    session.post(
                        args.url + "/generate",
                        json=dict(
                            text="2 + 2 =",
                            sampling_params=dict(temperature=0, max_new_tokens=1),
                        ),
                        timeout=120,
                    ).raise_for_status()
        finally:
            subprocess.run(["dcgmi", "profile", "--resume"], check=True)
    (args.output_dir / "complete.json").write_text(
        json.dumps(dict(completed=True, unix_s=time.time())) + "\n"
    )
    print("Audit capture complete", flush=True)


if __name__ == "__main__":
    main()
