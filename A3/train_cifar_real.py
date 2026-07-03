"""
Real (modest) CIFAR-10 training run for Assignment 3.

Trains a DeepConvNet (VGG-style, Kaiming init, Adam) on real CIFAR-10 and
reports measured train/val accuracy. CPU-only, kept small but real. Appends
measured numbers to ../results/a3_cifar.txt.

Run:
    OMP_NUM_THREADS=3 python train_cifar_real.py
"""
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.manual_seed(0)
torch.set_num_threads(3)

from eecs598.data import preprocess_cifar10
from eecs598 import Solver
from convolutional_networks import DeepConvNet
from fully_connected_networks import adam

device = "cpu"
dtype = torch.float32

# Modest but real: 10k train / 1k val subset, small VGG-style net.
data = preprocess_cifar10(cuda=False, dtype=dtype, flatten=False,
                          bias_trick=False, show_examples=False,
                          validation_ratio=0.1)
# Trim to keep CPU runtime modest.
small = {
    "X_train": data["X_train"][:10000],
    "y_train": data["y_train"][:10000],
    "X_val": data["X_val"][:1000],
    "y_val": data["y_val"][:1000],
}
print("train", tuple(small["X_train"].shape), "val", tuple(small["X_val"].shape))

model = DeepConvNet(
    input_dims=(3, 32, 32),
    num_filters=[32, 64, 128],
    max_pools=[0, 1, 2],
    weight_scale="kaiming",
    reg=1e-4,
    dtype=dtype,
    device=device,
)
solver = Solver(
    model, small,
    num_epochs=5,
    batch_size=128,
    update_rule=adam,
    optim_config={"learning_rate": 1e-3},
    print_every=40,
    device=device,
)
t0 = time.time()
solver.train()
elapsed = time.time() - t0

train_acc = solver.check_accuracy(small["X_train"], small["y_train"],
                                  num_samples=1000)
val_acc = solver.check_accuracy(small["X_val"], small["y_val"])
print(f"\nDeepConvNet CIFAR-10  train_acc(1k)={train_acc:.4f}  "
      f"val_acc={val_acc:.4f}  ({elapsed:.0f}s)")

res = {
    "deepconvnet_cifar_train_acc_1k": round(float(train_acc), 4),
    "deepconvnet_cifar_val_acc": round(float(val_acc), 4),
    "best_val_acc_during_training": round(float(solver.best_val_acc), 4),
    "train_seconds": round(elapsed, 1),
    "final_train_loss": round(float(solver.loss_history[-1]), 4),
}
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "a3_cifar.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A3 real CIFAR-10 DeepConvNet training\n")
    f.write("=" * 40 + "\n")
    f.write(json.dumps(res, indent=2))
print(f"Saved {out_path}")
