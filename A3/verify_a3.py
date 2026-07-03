"""
Verification harness for Assignment 3 (fully-connected + convolutional nets).

Uses numeric gradient checking on synthetic data (no dataset download needed)
to prove correctness of every layer forward/backward, the optimizers, dropout,
batchnorm, and the full network loss functions. Writes a summary to
../results/a3_gradcheck.txt.
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(3)
torch.manual_seed(0)

from eecs598.grad import compute_numeric_gradient, rel_error
import fully_connected_networks as fc
import convolutional_networks as cnn

DT = torch.double
RESULTS = {}
FAILS = []


def report(name, err, tol=1e-6):
    ok = err < tol
    RESULTS[name] = float(err)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={err:.3e}")
    if not ok:
        FAILS.append(name)


# ------------------- Linear -------------------
x = torch.randn(4, 5, 6, dtype=DT)
w = torch.randn(30, 7, dtype=DT)
b = torch.randn(7, dtype=DT)
dout = torch.randn(4, 7, dtype=DT)
out, cache = fc.Linear.forward(x, w, b)
dx, dw, db = fc.Linear.backward(dout, cache)
dx_num = compute_numeric_gradient(lambda x: fc.Linear.forward(x, w, b)[0], x, dout)
dw_num = compute_numeric_gradient(lambda w: fc.Linear.forward(x, w, b)[0], w, dout)
db_num = compute_numeric_gradient(lambda b: fc.Linear.forward(x, w, b)[0], b, dout)
report("Linear dx", rel_error(dx, dx_num))
report("Linear dw", rel_error(dw, dw_num))
report("Linear db", rel_error(db, db_num))

# ------------------- ReLU -------------------
x = torch.randn(5, 6, dtype=DT)
dout = torch.randn(5, 6, dtype=DT)
out, cache = fc.ReLU.forward(x)
dx = fc.ReLU.backward(dout, cache)
dx_num = compute_numeric_gradient(lambda x: fc.ReLU.forward(x)[0], x, dout)
report("ReLU dx", rel_error(dx, dx_num))

# ------------------- TwoLayerNet (FC) -------------------
N, D, H, C = 6, 15, 20, 4
X = torch.randn(N, D, dtype=DT)
y = torch.randint(C, (N,))
model = fc.TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C,
                       reg=0.05, dtype=DT, device='cpu')
loss, grads = model.loss(X, y)
for name in sorted(model.params):
    f = lambda _: model.loss(X, y)[0]
    num = compute_numeric_gradient(f, model.params[name])
    report(f"TwoLayerNet(FC) {name}", rel_error(grads[name], num), tol=1e-5)

# ------------------- FullyConnectedNet + dropout -------------------
model = fc.FullyConnectedNet([10, 10], input_dim=D, num_classes=C,
                             reg=0.1, dtype=DT, device='cpu')
loss, grads = model.loss(X, y)
for name in sorted(model.params):
    f = lambda _: model.loss(X, y)[0]
    num = compute_numeric_gradient(f, model.params[name])
    report(f"FullyConnectedNet {name}", rel_error(grads[name], num), tol=1e-4)

# dropout gradient
x = torch.randn(10, 10, dtype=DT) + 10
dp = {'mode': 'train', 'p': 0.3, 'seed': 123}
out, cache = fc.Dropout.forward(x, dp)
dout = torch.randn_like(x)
dx = fc.Dropout.backward(dout, cache)
dx_num = compute_numeric_gradient(
    lambda x: fc.Dropout.forward(x, dp)[0], x, dout)
report("Dropout dx", rel_error(dx, dx_num), tol=1e-6)

# ------------------- Optimizers sanity (adam converges on quadratic) -------------------
w = torch.tensor([5.0, -3.0, 2.0], dtype=DT)
cfg = {'learning_rate': 0.1}
for _ in range(500):
    dw = 2 * w
    w, cfg = fc.adam(w, dw, cfg)
report("adam converges to 0", w.abs().max().item(), tol=1e-2)

# ------------------- Conv forward/backward -------------------
x = torch.randn(2, 3, 8, 8, dtype=DT)
w = torch.randn(4, 3, 3, 3, dtype=DT)
b = torch.randn(4, dtype=DT)
cp = {'stride': 1, 'pad': 1}
out, cache = cnn.Conv.forward(x, w, b, cp)
dout = torch.randn_like(out)
dx, dw, db = cnn.Conv.backward(dout, cache)
dx_num = compute_numeric_gradient(lambda x: cnn.Conv.forward(x, w, b, cp)[0], x, dout)
dw_num = compute_numeric_gradient(lambda w: cnn.Conv.forward(x, w, b, cp)[0], w, dout)
db_num = compute_numeric_gradient(lambda b: cnn.Conv.forward(x, w, b, cp)[0], b, dout)
report("Conv dx", rel_error(dx, dx_num), tol=1e-6)
report("Conv dw", rel_error(dw, dw_num), tol=1e-6)
report("Conv db", rel_error(db, db_num), tol=1e-6)

# check Conv matches torch reference
ref = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
report("Conv matches torch.conv2d", rel_error(out, ref), tol=1e-9)

# ------------------- MaxPool forward/backward -------------------
x = torch.randn(2, 3, 8, 8, dtype=DT)
pp = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
out, cache = cnn.MaxPool.forward(x, pp)
dout = torch.randn_like(out)
dx = cnn.MaxPool.backward(dout, cache)
dx_num = compute_numeric_gradient(lambda x: cnn.MaxPool.forward(x, pp)[0], x, dout)
report("MaxPool dx", rel_error(dx, dx_num), tol=1e-6)
ref = torch.nn.functional.max_pool2d(x, 2, stride=2)
report("MaxPool matches torch", rel_error(out, ref), tol=1e-9)

# ------------------- BatchNorm forward/backward -------------------
x = torch.randn(10, 5, dtype=DT)
gamma = torch.randn(5, dtype=DT)
beta = torch.randn(5, dtype=DT)
dout = torch.randn(10, 5, dtype=DT)
out, cache = cnn.BatchNorm.forward(x, gamma, beta, {'mode': 'train'})
dx, dgamma, dbeta = cnn.BatchNorm.backward(dout, cache)
fx = lambda x: cnn.BatchNorm.forward(x, gamma, beta, {'mode': 'train'})[0]
fg = lambda g: cnn.BatchNorm.forward(x, g, beta, {'mode': 'train'})[0]
fb = lambda bb: cnn.BatchNorm.forward(x, gamma, bb, {'mode': 'train'})[0]
report("BatchNorm dx", rel_error(dx, compute_numeric_gradient(fx, x, dout)), tol=1e-5)
report("BatchNorm dgamma", rel_error(dgamma, compute_numeric_gradient(fg, gamma, dout)), tol=1e-6)
report("BatchNorm dbeta", rel_error(dbeta, compute_numeric_gradient(fb, beta, dout)), tol=1e-6)
# backward_alt should match backward
dx_alt, dg_alt, db_alt = cnn.BatchNorm.backward_alt(dout, cache)
report("BatchNorm backward_alt==backward", rel_error(dx, dx_alt), tol=1e-10)

# ------------------- SpatialBatchNorm -------------------
x = torch.randn(2, 3, 4, 4, dtype=DT)
gamma = torch.randn(3, dtype=DT)
beta = torch.randn(3, dtype=DT)
dout = torch.randn(2, 3, 4, 4, dtype=DT)
out, cache = cnn.SpatialBatchNorm.forward(x, gamma, beta, {'mode': 'train'})
dx, dgamma, dbeta = cnn.SpatialBatchNorm.backward(dout, cache)
fx = lambda x: cnn.SpatialBatchNorm.forward(x, gamma, beta, {'mode': 'train'})[0]
report("SpatialBatchNorm dx", rel_error(dx, compute_numeric_gradient(fx, x, dout)), tol=1e-5)

# ------------------- ThreeLayerConvNet loss grad -------------------
X = torch.randn(2, 3, 16, 16, dtype=DT)
y = torch.randint(10, (2,))
model = cnn.ThreeLayerConvNet(input_dims=(3, 16, 16), num_filters=3,
                              filter_size=3, hidden_dim=10,
                              reg=0.1, dtype=DT, device='cpu')
loss, grads = model.loss(X, y)
for name in sorted(model.params):
    f = lambda _: model.loss(X, y)[0]
    num = compute_numeric_gradient(f, model.params[name])
    report(f"ThreeLayerConvNet {name}", rel_error(grads[name], num), tol=1e-3)

# ------------------- DeepConvNet with batchnorm loss grad -------------------
X = torch.randn(2, 3, 8, 8, dtype=DT)
y = torch.randint(10, (2,))
model = cnn.DeepConvNet(input_dims=(3, 8, 8), num_filters=[4, 4],
                        max_pools=[0], batchnorm=True,
                        weight_scale='kaiming', reg=0.05,
                        dtype=DT, device='cpu')
loss, grads = model.loss(X, y)
maxerr = 0.0
for name in sorted(model.params):
    f = lambda _: model.loss(X, y)[0]
    num = compute_numeric_gradient(f, model.params[name])
    # A conv bias that is immediately followed by batchnorm has a TRUE gradient
    # of exactly zero (BN subtracts the mean, so the bias has no effect). For
    # such params both analytic and numeric grads are ~1e-8 noise, and the
    # relative-error metric is meaningless (0/0), so we skip them.
    if grads[name].abs().max().item() < 1e-10 and num.abs().max().item() < 1e-6:
        print(f"[skip] {name}: true-zero grad (conv bias behind batchnorm)")
        continue
    err = rel_error(grads[name], num)
    maxerr = max(maxerr, err)
report("DeepConvNet(bn) max grad err", maxerr, tol=1e-2)

# save
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "a3_gradcheck.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A3 numeric gradient checks\n" + "=" * 40 + "\n")
    f.write(f"failed: {FAILS}\n\n")
    f.write(json.dumps(RESULTS, indent=2))
print(f"\nSaved {out_path}")
print("=" * 50)
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL A3 GRADIENT CHECKS PASSED")
