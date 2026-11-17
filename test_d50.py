import os
import subprocess
import sys
import time
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plt.show() does not block
import torch

# Same launcher as test.py, for the game in D = 50 dimensions. Every run is its own process, with the
# environment variable DIM=50, and writes into out_<course>_d50.
DIM = 50
BATCH = 256           # the Laplacian trick repeats the batch DIM times, so a step costs about DIM / 2 times more
                      # than in 2D. 256 instead of 1024 keeps a step affordable. Raise it if your GPU is fast enough.
ITERS = 30000         # same as in 2D
FRESH_START = True    # set to False if you rerun this script to resume after an interruption

HAS_GPU = torch.cuda.is_available()
GPU_SLOTS = 4 if HAS_GPU else 0    # runs at the same time. At D = 50 a run keeps the GPU busy, so fewer fit than in 2D.
CPU_SLOTS = 0 if HAS_GPU else 3    # (raise GPU_SLOTS to 3 or 4 only if the it/s per run stays about the same)
print(f"GPU: {torch.cuda.get_device_name(0) if HAS_GPU else 'none'} | D = {DIM}, batch {BATCH}"
      f" | runs at the same time: {GPU_SLOTS} on the GPU + {CPU_SLOTS} on CPU cores")

RUNS = [("the_fork", "0"), ("the_fork", "1"), ("the_fork_hammers", "0"), ("the_fork_hammers", "1"),
        ("one_gap_pillar", "0"), ("one_gap_pillar", "1"), ("hammer_gates", "0"), ("hammer_gates", "1"), ("one_gap", "0"), ("one_gap", "1")]

def folder(c, seed):
    return f"out_{c}_d{DIM}" + ("" if seed == "0" else f"_seed{seed}")


if FRESH_START:
    for c, seed in RUNS:
        ckpt = os.path.join(folder(c, seed), "model.pt")
        if os.path.exists(ckpt):
            os.remove(ckpt)
            print("removed", ckpt)

t_all = time.time()
pending, running, failed = list(RUNS), [], []
free = ["cuda"] * GPU_SLOTS + ["cpu"] * CPU_SLOTS
while pending or running:
    while pending and free:
        (c, seed), dev = pending.pop(0), free.pop(0)
        os.makedirs(folder(c, seed), exist_ok=True)
        env = dict(os.environ, COURSE=c, SEED=seed, RETRAIN="0", MPLBACKEND="Agg", DEVICE=dev,
                   DIM=str(DIM), BATCH=str(BATCH), ITERS=str(ITERS))
        if dev == "cpu":
            env.update(THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        console = open(os.path.join(folder(c, seed), "console.txt"), "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "apac_fallguys_v3.py"], env=env,
                                stdout=console, stderr=subprocess.STDOUT)
        running.append((c, seed, dev, proc, console, time.time()))
        print(f"started  {c} seed {seed} on {dev}  (console output: {folder(c, seed)}/console.txt)")
    for item in list(running):
        c, seed, dev, proc, console, t0 = item
        if proc.poll() is not None:
            console.close()
            running.remove(item)
            free.insert(0, dev) if dev == "cuda" else free.append(dev)
            status = "done" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
            if proc.returncode != 0:
                failed.append((c, seed))
            print(f"{status:8s} {c} seed {seed} on {dev}  after {(time.time() - t0) / 60:.1f} min")
    time.sleep(5)

if failed:
    print("failed runs (see their console.txt):", failed)
print(f"all runs finished after {(time.time() - t_all) / 60:.1f} min")

# Unified report over the finished D = 50 runs: summary.txt (table), summary.csv, summary.png
# (written to the folder summary_d50 so it does not overwrite the 2D summary)
import apac_fallguys_v3 as apac
os.makedirs("summary_d50", exist_ok=True)
for c, seed in RUNS:
    src = os.path.join(folder(c, seed), "metrics.json")
    if os.path.exists(src):
        dst = os.path.join("summary_d50", os.path.basename(folder(c, seed)))
        os.makedirs(dst, exist_ok=True)
        with open(src, encoding="utf-8") as fi, open(os.path.join(dst, "metrics.json"), "w", encoding="utf-8") as fo:
            fo.write(fi.read())
apac.write_summary("summary_d50")
