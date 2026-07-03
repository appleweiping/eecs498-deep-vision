"""
Verification harness for Assignment 1 (PyTorch 101 + kNN).

Runs correctness checks for every implemented function in pytorch101.py and
knn.py, then trains/evaluates a real kNN classifier on CIFAR-10 (downloaded at
runtime) and runs 5-fold cross-validation. Results are printed and saved to
../results/a1_knn.txt.

Run:
    OMP_NUM_THREADS=3 python verify_a1.py
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(__file__))

import pytorch101
import knn
from eecs598.data import cifar10

torch.manual_seed(0)
torch.set_num_threads(3)

FAILS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        FAILS.append(name)


def rel_err(a, b):
    return (a - b).abs().max().item()


# ----------------------- pytorch101 checks -----------------------------------
print("=== pytorch101.py ===")
x = pytorch101.create_sample_tensor()
check("create_sample_tensor", x.shape == (3, 2) and x[0, 1] == 10 and x[1, 0] == 100)

t = torch.zeros(2, 2)
pytorch101.mutate_tensor(t, [(0, 0), (1, 1), (0, 0)], [1.0, 2.0, 3.0])
check("mutate_tensor", t[0, 0] == 3 and t[1, 1] == 2)

check("count_tensor_elements", pytorch101.count_tensor_elements(torch.zeros(2, 3, 4)) == 24)
check("create_tensor_of_pi", torch.allclose(pytorch101.create_tensor_of_pi(2, 3), torch.full((2, 3), 3.14)))

mt = pytorch101.multiples_of_ten(5, 45)
check("multiples_of_ten", mt.tolist() == [10.0, 20.0, 30.0, 40.0] and mt.dtype == torch.float64)
check("multiples_of_ten empty", pytorch101.multiples_of_ten(3, 7).shape == (0,))

xs = torch.arange(30).reshape(5, 6)
lr, tc, ftr, erc = pytorch101.slice_indexing_practice(xs)
check("slice_indexing last_row", lr.tolist() == [24, 25, 26, 27, 28, 29])
check("slice_indexing third_col", tc.squeeze().tolist() == [2, 8, 14, 20, 26])
check("slice_indexing first_two_rows_three_cols", ftr.tolist() == [[0, 1, 2], [6, 7, 8]])

sa = pytorch101.slice_assignment_practice(torch.zeros(4, 6, dtype=torch.long))
expected = torch.tensor([[0, 1, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2],
                         [3, 4, 3, 4, 5, 5], [3, 4, 3, 4, 5, 5]])
check("slice_assignment_practice", torch.equal(sa, expected))

sc = pytorch101.shuffle_cols(torch.tensor([[1, 2, 3], [4, 5, 6]]))
check("shuffle_cols", sc.tolist() == [[1, 1, 3, 2], [4, 4, 6, 5]])

rr = pytorch101.reverse_rows(torch.tensor([[1, 2], [3, 4], [5, 6]]))
check("reverse_rows", rr.tolist() == [[5, 6], [3, 4], [1, 2]])

toe = pytorch101.take_one_elem_per_col(torch.arange(12).reshape(4, 3))
check("take_one_elem_per_col", toe.tolist() == [3, 1, 11])

oh = pytorch101.make_one_hot([1, 4, 3, 2])
check("make_one_hot", oh.shape == (4, 5) and oh[0, 1] == 1 and oh[1, 4] == 1 and oh.dtype == torch.float32)

check("sum_positive_entries",
      pytorch101.sum_positive_entries(torch.tensor([[-1, 2, 0], [0, 5, -3], [8, -9, 0]])) == 15)

rp = pytorch101.reshape_practice(torch.arange(24))
check("reshape_practice", rp.tolist()[0] == [0, 1, 2, 3, 12, 13, 14, 15])

zr = pytorch101.zero_row_min(torch.tensor([[10, 20, 30], [2, 5, 1]]))
check("zero_row_min", zr.tolist() == [[0, 20, 30], [2, 5, 0]])

xb = torch.randn(4, 3, 5)
yb = torch.randn(4, 5, 2)
zl = pytorch101.batched_matrix_multiply(xb, yb, use_loop=True)
zn = pytorch101.batched_matrix_multiply(xb, yb, use_loop=False)
ref = torch.bmm(xb, yb)
check("batched_matrix_multiply_loop", rel_err(zl, ref) < 1e-5)
check("batched_matrix_multiply_noloop", rel_err(zn, ref) < 1e-5)

xn = torch.randn(50, 8)
yn = pytorch101.normalize_columns(xn)
ref_norm = (xn - xn.mean(0)) / xn.std(0)
check("normalize_columns", rel_err(yn, ref_norm) < 1e-4)

xg = torch.randn(10, 4)
wg = torch.randn(4, 6)
check("mm_on_gpu(cpu fallback)", rel_err(pytorch101.mm_on_gpu(xg, wg), xg.mm(wg)) < 1e-5)

xs_list = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0]), torch.tensor([10.0])]
ls = torch.tensor([3, 2, 1])
means = pytorch101.challenge_mean_tensors(xs_list, ls)
check("challenge_mean_tensors", torch.allclose(means, torch.tensor([2.0, 5.0, 10.0])))

xu = torch.tensor([3, 1, 3, 2, 1, 5, 2])
uniq, idx = pytorch101.challenge_get_uniques(xu)
check("challenge_get_uniques values", torch.equal(uniq, torch.unique(xu)))
check("challenge_get_uniques first-occurrence", torch.equal(xu[idx], uniq))


# ----------------------- knn checks ------------------------------------------
print("\n=== knn.py distance-function agreement ===")
xt = torch.randn(20, 3, 4, 4)
xe = torch.randn(7, 3, 4, 4)
d2 = knn.compute_distances_two_loops(xt, xe)
d1 = knn.compute_distances_one_loop(xt, xe)
d0 = knn.compute_distances_no_loops(xt, xe)
# reference brute force
ref = ((xt.reshape(20, -1)[:, None, :] - xe.reshape(7, -1)[None, :, :]) ** 2).sum(-1)
check("two_loops vs ref", rel_err(d2, ref) < 1e-3)
check("one_loop vs two_loop", rel_err(d1, d2) < 1e-4)
check("no_loop vs two_loop", rel_err(d0, d2) < 1e-2)

dists = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
y_train = torch.tensor([1, 2, 1, 2, 3])
pred = knn.predict_labels(dists, y_train, k=5)
check("predict_labels tie->smallest", pred.tolist() == [1, 1])


# ----------------------- REAL run on CIFAR-10 --------------------------------
print("\n=== Real kNN on CIFAR-10 ===")
num_train = 5000
num_test = 500
x_train, y_train, x_test, y_test = cifar10(num_train, num_test)
print(f"train {tuple(x_train.shape)}  test {tuple(x_test.shape)}")

results = {}
clf = knn.KnnClassifier(x_train, y_train)
for k in [1, 5]:
    acc = clf.check_accuracy(x_test, y_test, k=k)
    results[f"knn_k{k}_test_acc"] = acc

print("\n=== 5-fold cross-validation (subset) ===")
k_choices = [1, 3, 5, 8, 12, 20]
k_to_acc = knn.knn_cross_validate(x_train, y_train, num_folds=5, k_choices=k_choices)
cv_summary = {}
for k in k_choices:
    accs = k_to_acc[k]
    mean = sum(accs) / len(accs)
    cv_summary[k] = round(mean, 2)
    print(f"k={k:3d}  mean CV acc = {mean:.2f}%")
best_k = knn.knn_get_best_k(k_to_acc)
print(f"best k = {best_k}")
results["cv_mean_acc_by_k"] = cv_summary
results["best_k"] = best_k

# final accuracy with best k
best_acc = clf.check_accuracy(x_test, y_test, k=best_k)
results[f"knn_best_k{best_k}_test_acc"] = best_acc

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "a1_knn.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A1 verification\n")
    f.write("=" * 40 + "\n")
    f.write(f"correctness checks failed: {FAILS}\n\n")
    f.write(json.dumps(results, indent=2))
    f.write("\n")
print(f"\nSaved results to {out_path}")

print("\n" + "=" * 50)
if FAILS:
    print(f"FAILED CHECKS: {FAILS}")
    sys.exit(1)
else:
    print("ALL A1 CORRECTNESS CHECKS PASSED")
