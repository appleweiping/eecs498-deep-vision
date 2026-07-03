"""
Verification harness for Assignment 2 (SVM / Softmax linear classifiers +
two-layer net). Uses numeric gradient checking on synthetic data and confirms
the naive and vectorized implementations agree. Writes a summary to
../results/a2_gradcheck.txt.
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(3)
torch.manual_seed(0)

from eecs598.grad import compute_numeric_gradient, rel_error
import linear_classifier as lc
import two_layer_net as tln

RESULTS = {}
FAILS = []


def scalar(x):
    return x.item() if torch.is_tensor(x) else float(x)


def report(name, err, tol=1e-6):
    ok = err < tol
    RESULTS[name] = float(err)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={err:.3e}")
    if not ok:
        FAILS.append(name)


DT = torch.double
D, C, N = 20, 5, 30
W = torch.randn(D, C, dtype=DT) * 0.01
X = torch.randn(N, D, dtype=DT)
y = torch.randint(C, (N,))
reg = 0.05

# ---- SVM: naive vs vectorized ----
loss_n, dW_n = lc.svm_loss_naive(W, X, y, reg)
loss_v, dW_v = lc.svm_loss_vectorized(W, X, y, reg)
report("SVM loss naive==vectorized", abs(scalar(loss_n) - scalar(loss_v)), tol=1e-8)
report("SVM dW naive==vectorized", rel_error(dW_n, dW_v), tol=1e-8)
# numeric grad check
f = lambda w: lc.svm_loss_vectorized(w, X, y, reg)[0]
dW_num = compute_numeric_gradient(f, W)
report("SVM dW numeric", rel_error(dW_v, dW_num), tol=1e-5)

# ---- Softmax: naive vs vectorized ----
loss_n, dW_n = lc.softmax_loss_naive(W, X, y, reg)
loss_v, dW_v = lc.softmax_loss_vectorized(W, X, y, reg)
report("Softmax loss naive==vectorized", abs(scalar(loss_n) - scalar(loss_v)), tol=1e-8)
report("Softmax dW naive==vectorized", rel_error(dW_n, dW_v), tol=1e-8)
f = lambda w: lc.softmax_loss_vectorized(w, X, y, reg)[0]
dW_num = compute_numeric_gradient(f, W)
report("Softmax dW numeric", rel_error(dW_v, dW_num), tol=1e-5)
# softmax loss on random weights should be ~ log(C)
import math
loss0, _ = lc.softmax_loss_vectorized(W, X, y, 0.0)
report("Softmax init loss ~ log(C)", abs(scalar(loss0) - math.log(C)), tol=0.5)

# ---- Two-layer net (A2 explicit backprop) ----
D2, H2, C2, N2 = 12, 15, 4, 20
X2 = torch.randn(N2, D2, dtype=DT)
y2 = torch.randint(C2, (N2,))
net = tln.TwoLayerNet(D2, H2, C2, dtype=DT, device='cpu', std=1e-1)
loss, grads = tln.nn_forward_backward(net.params, X2, y2, reg=0.05)
for name in sorted(net.params):
    f = lambda _: tln.nn_forward_backward(net.params, X2, y2, reg=0.05)[0]
    num = compute_numeric_gradient(f, net.params[name])
    report(f"TwoLayerNet(A2) {name}", rel_error(grads[name], num), tol=1e-5)

# save
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "a2_gradcheck.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A2 numeric gradient checks\n" + "=" * 40 + "\n")
    f.write(f"failed: {FAILS}\n\n")
    f.write(json.dumps(RESULTS, indent=2))
print(f"\nSaved {out_path}")
print("=" * 50)
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL A2 GRADIENT CHECKS PASSED")
