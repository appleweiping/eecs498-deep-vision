"""
Verification harness for Assignment 5 (RNN/LSTM/attention captioning +
Transformer).

- numeric gradient checks for the vanilla-RNN building blocks
- forward-consistency checks for LSTM / attention / transformer components
- a REAL (modest) Transformer training run on the two-digit add/subtract toy
  dataset, reporting measured validation accuracy.

Writes ../results/a5_transformer.txt and a5_transformer_loss.png.
"""
import os
import sys
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
torch.manual_seed(0)
torch.set_num_threads(3)

from eecs598.grad import compute_numeric_gradient, rel_error
import rnn_lstm_captioning as rnn
import transformers as tf

RESULTS = {}
FAILS = []
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES_DIR, exist_ok=True)


def report(name, err, tol=1e-6):
    ok = err < tol if isinstance(err, float) else err
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f": {err:.3e}" if isinstance(err, float) else ""))
    if not ok:
        FAILS.append(name)


DT = torch.double

# ---------------- RNN step numeric grad ----------------
N, D, H = 3, 10, 4
x = torch.randn(N, D, dtype=DT)
prev_h = torch.randn(N, H, dtype=DT)
Wx = torch.randn(D, H, dtype=DT)
Wh = torch.randn(H, H, dtype=DT)
b = torch.randn(H, dtype=DT)
out, cache = rnn.rnn_step_forward(x, prev_h, Wx, Wh, b)
dnext = torch.randn(N, H, dtype=DT)
dx, dph, dWx, dWh, db = rnn.rnn_step_backward(dnext, cache)
fx = lambda x: rnn.rnn_step_forward(x, prev_h, Wx, Wh, b)[0]
fWx = lambda W: rnn.rnn_step_forward(x, prev_h, W, Wh, b)[0]
report("rnn_step dx", rel_error(dx, compute_numeric_gradient(fx, x, dnext)), 1e-7)
report("rnn_step dWx", rel_error(dWx, compute_numeric_gradient(fWx, Wx, dnext)), 1e-7)

# ---------------- RNN full-sequence numeric grad ----------------
T = 5
x = torch.randn(N, T, D, dtype=DT)
h0 = torch.randn(N, H, dtype=DT)
h, cache = rnn.rnn_forward(x, h0, Wx, Wh, b)
dh = torch.randn(N, T, H, dtype=DT)
dx, dh0, dWx, dWh, db = rnn.rnn_backward(dh, cache)
fx = lambda x: rnn.rnn_forward(x, h0, Wx, Wh, b)[0]
fh0 = lambda h0: rnn.rnn_forward(x, h0, Wx, Wh, b)[0]
report("rnn_forward dx", rel_error(dx, compute_numeric_gradient(fx, x, dh)), 1e-6)
report("rnn_forward dh0", rel_error(dh0, compute_numeric_gradient(fh0, h0, dh)), 1e-6)

# ---------------- WordEmbedding & temporal loss ----------------
V = 8
we = rnn.WordEmbedding(V, D)
idx = torch.randint(V, (N, T))
emb = we(idx)
report("WordEmbedding shape", emb.shape == (N, T, D))
scores = torch.randn(N, T, V)
y = torch.randint(V, (N, T))
loss = rnn.temporal_softmax_loss(scores, y, ignore_index=0)
report("temporal_softmax_loss finite", torch.isfinite(loss).item())

# ---------------- LSTM & attention forward ----------------
lstm = rnn.LSTM(D, H)
hn = lstm(torch.randn(N, T, D), torch.randn(N, H))
report("LSTM forward shape", hn.shape == (N, T, H))
A = torch.randn(N, H, 4, 4)
attn, aw = rnn.dot_product_attention(torch.randn(N, H), A)
report("attention shapes", attn.shape == (N, H) and aw.shape == (N, 4, 4))
# attention weights sum to 1
report("attention weights sum to 1",
       torch.allclose(aw.reshape(N, -1).sum(1), torch.ones(N), atol=1e-5))
attn_lstm = rnn.AttentionLSTM(D, H)
hn = attn_lstm(torch.randn(N, T, D), A)
report("AttentionLSTM forward shape", hn.shape == (N, T, H))

# ---------------- Transformer components ----------------
print("\n=== Transformer components ===")
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)
# no-loop should match two-loop
y_loop = tf.scaled_dot_product_two_loop_batch(q, k, v)
y_noloop, w = tf.scaled_dot_product_no_loop_batch(q, k, v)
report("attn two-loop == no-loop", rel_error(y_loop, y_noloop), 1e-5)
report("softmax weights sum to 1",
       torch.allclose(w.sum(dim=2), torch.ones(2, 4), atol=1e-5))

# masked attention: future positions must not leak
mask = tf.get_subsequent_mask(torch.zeros(2, 4))
_, wm = tf.scaled_dot_product_no_loop_batch(q, k, v, mask)
report("causal mask zeros future", float(wm[0, 0, 1:].sum().item()) < 1e-6, tol=1e-6)

# LayerNorm normalizes
ln = tf.LayerNormalization(8)
xn = ln(torch.randn(2, 4, 8))
report("LayerNorm ~zero mean", float(xn.mean().abs().item()) < 0.3, tol=1)

# full encoder/decoder block shape
enc = tf.EncoderBlock(num_heads=2, emb_dim=8, feedforward_dim=16, dropout=0.0)
eo = enc(torch.randn(2, 4, 8))
report("EncoderBlock shape", eo.shape == (2, 4, 8))
dec = tf.DecoderBlock(num_heads=2, emb_dim=8, feedforward_dim=16, dropout=0.0)
do = dec(torch.randn(2, 4, 8), eo, mask)
report("DecoderBlock shape", do.shape == (2, 4, 8))

# positional encodings
pe = tf.position_encoding_sinusoid(4, 8)
report("pos encoding shape", pe.shape == (1, 4, 8))

# ---------------- REAL Transformer training on add/subtract toy ----------------
print("\n=== REAL Transformer training (two-digit add/subtract) ===")
import json as _json
from torch.utils.data import DataLoader
from a5_helper import train as tf_train, val as tf_val

data = _json.load(open(os.path.join(os.path.dirname(__file__), "two_digit_op.json")))
inp_exp = data["inp_expression"]
out_exp = data["out_expression"]

vocab = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "BOS", "EOS", "POSITIVE", "NEGATIVE", "add", "subtract",
]
token_dict = tf.generate_token_dict(vocab)
special = {"BOS": token_dict["BOS"], "EOS": token_dict["EOS"],
           "POSITIVE": token_dict["POSITIVE"], "NEGATIVE": token_dict["NEGATIVE"],
           "add": token_dict["add"], "subtract": token_dict["subtract"]}

EMB = 32
pos_fn = tf.position_encoding_sinusoid
# use a subset for a CPU-modest but real run
n_total = len(inp_exp)
n_train = min(4000, int(0.8 * n_total))
n_val = min(1000, n_total - n_train)
train_ds = tf.AddSubDataset(inp_exp[:n_train], out_exp[:n_train], token_dict,
                            special, EMB, pos_fn)
val_ds = tf.AddSubDataset(inp_exp[n_train:n_train + n_val],
                          out_exp[n_train:n_train + n_val], token_dict,
                          special, EMB, pos_fn)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = tf.Transformer(num_heads=4, emb_dim=EMB, feedforward_dim=64,
                       dropout=0.1, num_enc_layers=2, num_dec_layers=2,
                       vocab_len=len(vocab))
loss_func = torch.nn.CrossEntropyLoss(reduction="sum")

model = tf_train(model, train_loader, val_loader, loss_func, num_epochs=16,
                 batch_size=64, warmup_lr=1e-4, warmup_interval=200, lr=8e-4)


@torch.no_grad()
def token_accuracy(model, loader):
    """Correctly-accumulated token accuracy over the whole val set (the
    official a5_helper.val() only counts the last batch, so we compute it
    here)."""
    model.eval()
    correct, total = 0, 0
    for inp, inp_pos, out, out_pos in loader:
        gnd = out[:, 1:].contiguous().view(-1).long()
        pred = model(inp.long(), inp_pos, out.long(), out_pos)
        pred_max = pred.max(1)[1]
        correct += pred_max.eq(gnd).sum().item()
        total += gnd.numel()
    return correct / total


val_acc = token_accuracy(model, val_loader)
val_loss, _ = tf_val(model, val_loader, loss_func, 64)
RESULTS["transformer_val_loss"] = round(float(val_loss), 4)
RESULTS["transformer_val_token_acc"] = round(float(val_acc), 4)
print(f"Transformer val loss={val_loss:.4f} token-acc={val_acc:.4f}")
report("Transformer learns (val token acc > 0.5)", float(val_acc) > 0.5, tol=1)

# ---------------- save ----------------
out_path = os.path.join(RES_DIR, "a5_transformer.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A5 (RNN + Transformer) verification\n" + "=" * 40 + "\n")
    f.write(f"failed checks: {FAILS}\n\n")
    f.write(json.dumps(RESULTS, indent=2))
print(f"\nSaved {out_path}")
print("=" * 50)
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL A5 CHECKS PASSED")
