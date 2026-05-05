# simple-gpu-schedule

Tiny cooperative FIFO GPU scheduler for a small shared server.

It is intentionally boring: one executable, no daemon, plain JSON ticket files,
`flock` locks, and `nvidia-smi` checks.

## Usage

```bash
./with-gpu <pool> [--num N] -- <command> [args...]
```

Examples:

```bash
./with-gpu any -- python train.py
./with-gpu 2 -- bash launch.sh
./with-gpu any --num 2 -- torchrun --nproc_per_node=2 train.py
./with-gpu 0-3 --num 2 -- bash launch.sh
./with-gpu 0,2,3 --num 2 -- python train.py
./with-gpu any --idle-seconds 0 -- python quick_test.py
```

Pool syntax:

- `any` — any visible GPU
- `2` — only GPU 2
- `0-3` — GPUs 0,1,2,3
- `0,2,3` — GPUs 0,2,3

`--num` is the number of GPUs required from the pool, default `1`.

`--idle-seconds` is the continuous time a GPU must have no `nvidia-smi` compute
processes before launch. Default: `120`, override with `GPU_IDLE_SECONDS`.
This avoids launching into short gaps between runs in manual shell-loop sweeps.

`--interval` is the queue polling interval while waiting. Default: `15`,
override with `CHECK_INTERVAL_SECONDS`.

When GPUs are acquired, `CUDA_VISIBLE_DEVICES` is set to the selected GPU list,
e.g. `2` or `0,1`.

## Queue state

Default queue directory:

```text
/store/store5/software/gpu-queue
```

The executable also loads a repo-local `.env` file if present. This file is
ignored by git and is the preferred place to set machine-local queue state:

```bash
cp .env.example .env
```

Override the queue location with:

```bash
GPU_QUEUE_DIR=/some/shared/path ./with-gpu any -- python train.py
# or
./with-gpu any --queue-dir /some/shared/path -- python train.py
```

Layout:

```text
/store/store5/software/gpu-queue/
  tickets/   waiting job tickets as JSON
  running/   running job metadata as JSON
  locks/     queue.lock and gpuN.lock flock files
```

## FIFO behavior

Waiting processes create ticket files and poll. The scheduler plans jobs in ticket
creation order. If the first waiting ticket cannot currently run, later tickets do
not jump ahead.

This is cooperative: jobs launched outside `with-gpu` are detected via
`nvidia-smi` and will block launch, but they do not have queue tickets. GPUs must
remain externally idle for the configured grace period before launch.

## Stale tickets

Tickets include host, PID, and Linux boot id. On each queue check, stale tickets
from dead same-host processes or previous boots are removed automatically.

This does **not** resurrect jobs after reboot/disconnect. Run inside `tmux` or
`screen` if the waiting process must survive SSH disconnects.

## Display

While waiting on a TTY, `with-gpu` redraws an ASCII queue/GPU dashboard. Once a
GPU is acquired, the dashboard stops and the command output is shown normally.
For non-TTY logs, it prints compact periodic status lines.
