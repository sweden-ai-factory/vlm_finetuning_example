# tools/check_env.py
"""
Print what the job actually got: software versions, GPUs, CPU cores, caches.
Called at the top of every slurm script so that the log file documents the
environment. Also handy to run by hand when something does not work.
"""

import os
import platform
import sys

print("host:", platform.node(), "| python:", sys.version.split()[0])
print("CPU cores available to this job:", len(os.sched_getaffinity(0)))
for var in ["SLURM_JOB_ID", "SLURM_GPUS", "SLURM_CPUS_PER_TASK", "HF_HOME", "HF_HUB_OFFLINE"]:
    print(f"{var} = {os.environ.get(var)}")

for name in ["torch", "torchvision", "transformers", "accelerate", "peft", "datasets", "PIL"]:
    try:
        mod = __import__(name)
        print(f"{name:13s} {mod.__version__}")
    except Exception as e:
        print(f"{name:13s} MISSING ({e})")

import torch

print("GPU available:", torch.cuda.is_available(), "| count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)} ({mem:.0f} GB)")
